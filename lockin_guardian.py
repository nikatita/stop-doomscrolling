"""
Lock-In Guardian
=================

Watches your webcam. If it thinks you've been looking down (at your phone,
presumably doomscrolling) continuously for too long, it pauses and plays
your "lock in" video to snap you back to focus.

HOW IT DETECTS "DOOMSCROLLING"
-------------------------------
Two signals have to agree before this calls it "doomscrolling":

  1. HEAD POSE: MediaPipe FaceMesh + solvePnP estimates your head pitch.
     If it's tilted down past DOWN_PITCH_THRESHOLD_DEG, that's "looking down".
  2. PHONE PRESENCE: A small YOLOv8 object detector (trained on COCO, which
     includes a "cell phone" class) checks whether an actual phone is
     visible in frame.

Only when BOTH are true, continuously (with small tolerance for blinks/
occlusion), for DOOMSCROLL_SECONDS, do we trigger the video. This cuts out
a lot of false positives (e.g. reading a physical book, looking down at a
laptop with no phone in view).

Set ENABLE_PHONE_DETECTION = False if you'd rather go back to head-pose-only
(faster, but more false positives).

This is still a proxy, not mind-reading — it can't tell *which app* you're
in. Tune the thresholds below to taste.

SETUP
-----
    pip install opencv-python mediapipe numpy ultralytics

First run will auto-download the yolov8n.pt weights (~6MB) from Ultralytics'
servers, so you'll need internet access once.

Then edit the CONFIG section below (especially VIDEO_PATH) and run:

    python lockin_guardian.py

Press 'q' in the webcam window to quit.
"""

import cv2
import time
import os
import sys
import platform
import subprocess
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
#enter yer vid lol
VIDEO_PATH = "C:/Users/Nikita/Downloads/BITCH_LOCK_IN.mp4"
DOOMSCROLL_SECONDS = 1.0
DOWN_PITCH_THRESHOLD_DEG = 7.0
FACE_LOST_GRACE_SECONDS = 2.0
COOLDOWN_SECONDS = 15.0
SHOW_DEBUG_OVERLAY = True
CAMERA_INDEX = 0
ENABLE_PHONE_DETECTION = True
YOLO_MODEL = "yolov8n.pt"
PHONE_CONFIDENCE_THRESHOLD = 0.45
PHONE_DETECT_EVERY_N_FRAMES = 3
PHONE_DETECTION_GRACE_SECONDS = 2.5
PHONE_CLASS_NAME = "cell phone"
mp_face_mesh = mp.solutions.face_mesh
LANDMARK_IDS = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_corner": 33,
    "right_eye_corner": 263,
    "left_mouth": 61,
    "right_mouth": 291,
}

MODEL_POINTS_3D = np.array(
    [
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0),
    ],
    dtype=np.float64,
)


def estimate_head_pose(landmarks, frame_w, frame_h):
    """
    Given MediaPipe FaceMesh landmarks, estimate (pitch, yaw, roll) in degrees.
    Pitch > 0 roughly means looking down (sign convention handled below).
    Returns None if pose can't be estimated.
    """
    try:
        image_points = np.array(
            [
                (landmarks[LANDMARK_IDS["nose_tip"]].x * frame_w,
                 landmarks[LANDMARK_IDS["nose_tip"]].y * frame_h),
                (landmarks[LANDMARK_IDS["chin"]].x * frame_w,
                 landmarks[LANDMARK_IDS["chin"]].y * frame_h),
                (landmarks[LANDMARK_IDS["left_eye_corner"]].x * frame_w,
                 landmarks[LANDMARK_IDS["left_eye_corner"]].y * frame_h),
                (landmarks[LANDMARK_IDS["right_eye_corner"]].x * frame_w,
                 landmarks[LANDMARK_IDS["right_eye_corner"]].y * frame_h),
                (landmarks[LANDMARK_IDS["left_mouth"]].x * frame_w,
                 landmarks[LANDMARK_IDS["left_mouth"]].y * frame_h),
                (landmarks[LANDMARK_IDS["right_mouth"]].x * frame_w,
                 landmarks[LANDMARK_IDS["right_mouth"]].y * frame_h),
            ],
            dtype=np.float64,
        )

        focal_length = frame_w
        center = (frame_w / 2, frame_h / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1))  

        success, rotation_vec, _translation_vec = cv2.solvePnP(
            MODEL_POINTS_3D, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return None

        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        proj_matrix = np.hstack((rotation_mat, np.zeros((3, 1))))
        euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)[6]
        pitch, yaw, roll = [float(a) for a in euler_angles.flatten()]
        return pitch, yaw, roll
    except Exception:
        return None


def normalize_pitch(pitch):
    """
    cv2's decomposeProjectionMatrix can return pitch wrapped around +-180.
    Normalize so that positive values mean 'looking down'.
    """
    if pitch < -90:
        pitch = -(180 + pitch)
    elif pitch > 90:
        pitch = 180 - pitch
    return -pitch  


def detect_phone(frame, yolo_model):
    """
    Runs YOLOv8 on the frame and returns True if a "cell phone" is detected
    above PHONE_CONFIDENCE_THRESHOLD.
    """
    try:
        results = yolo_model(frame, verbose=False)
        if not results:
            return False
        result = results[0]
        names = result.names 
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if names.get(cls_id) == PHONE_CLASS_NAME and conf >= PHONE_CONFIDENCE_THRESHOLD:
                return True
        return False
    except Exception as e:
        print(f"[!] Phone detection error: {e}")
        return False


def play_lockin_video(video_path):
    if not os.path.exists(video_path):
        print(f"[!] Video not found at '{video_path}'. Skipping playback, "
              f"but here's your reminder: GET BACK TO WORK.")
        time.sleep(3)
        return

    system = platform.system()
    print(f"[*] Doomscrolling detected. Launching lock-in video: {video_path}")

    try:
        if system == "Windows":
            vlc_path = r"C:\Program Files (x86)\VideoLAN\VLC\vlc.exe"
            process = subprocess.Popen([
        vlc_path,
        "--fullscreen",
        "--play-and-exit",
        "--no-video-title-show",
        os.path.abspath(video_path)
    ])
            process.wait()
            return

        elif system == "Darwin":
            subprocess.run(["open", video_path])

        else:  
            subprocess.run(["xdg-open", video_path])

    except Exception as e:
        print(f"[!] Could not open video automatically ({e}). "
            f"Open it yourself: {video_path}")
    duration = get_video_duration_seconds(video_path)
    wait_time = min(max(duration, 5), 600)
    print(f"[*] Waiting ~{wait_time:.0f}s for video to play out...")
    time.sleep(wait_time)


def get_video_duration_seconds(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 15.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or (fps * 15)
    cap.release()
    if fps <= 0:
        return 15.0
    return frame_count / fps


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[!] Could not open webcam. Check CAMERA_INDEX / permissions.")
        sys.exit(1)

    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    yolo_model = None
    if ENABLE_PHONE_DETECTION:
        print(f"[*] Loading phone detector ({YOLO_MODEL})... "
              f"first run may download weights.")
        yolo_model = YOLO(YOLO_MODEL)

    down_since = None         
    last_face_seen = time.time()
    cooldown_until = 0.0
    last_phone_seen = 0.0      
    frame_count = 0

    print("[*] Lock-In Guardian running. Press 'q' in the window to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[!] Failed to read frame from webcam.")
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        now = time.time()
        in_cooldown = now < cooldown_until
        status_text = "OK"
        pitch_display = None
        frame_count += 1

        phone_currently_visible = False
        if ENABLE_PHONE_DETECTION and not in_cooldown:
            if frame_count % PHONE_DETECT_EVERY_N_FRAMES == 0:
                phone_currently_visible = detect_phone(frame, yolo_model)
                if phone_currently_visible:
                    last_phone_seen = now
            phone_recently_seen = (now - last_phone_seen) <= PHONE_DETECTION_GRACE_SECONDS
        else:
            phone_recently_seen = True

        if results.multi_face_landmarks and not in_cooldown:
            landmarks = results.multi_face_landmarks[0].landmark
            pose = estimate_head_pose(landmarks, w, h)
            last_face_seen = now

            if pose is not None:
                raw_pitch, yaw, roll = pose
                pitch = normalize_pitch(raw_pitch)
                pitch_display = pitch

                looking_down = pitch > DOWN_PITCH_THRESHOLD_DEG

                if looking_down and phone_recently_seen:
                    if down_since is None:
                        down_since = now
                    elapsed = now - down_since
                    status_text = f"LOOKING DOWN + PHONE ({elapsed:.1f}s)"

                    if elapsed >= DOOMSCROLL_SECONDS:
                        cv2.putText(frame, "LOCK IN TIME", (20, h - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        cv2.imshow("Lock-In Guardian", frame)
                        cv2.waitKey(1)

                        play_lockin_video(VIDEO_PATH)

                        down_since = None
                        cooldown_until = time.time() + COOLDOWN_SECONDS
                elif looking_down and not phone_recently_seen:
                    down_since = None
                    status_text = "looking down (no phone)"
                else:
                    down_since = None
                    status_text = "OK (heads up)"
        elif not in_cooldown:
            
            if down_since is not None:
                if now - last_face_seen > FACE_LOST_GRACE_SECONDS:
                    down_since = None
                    status_text = "no face (reset)"
                else:
                    status_text = "no face (grace period)"
            else:
                status_text = "no face"
        else:
            status_text = f"cooldown ({cooldown_until - now:.0f}s)"

        if SHOW_DEBUG_OVERLAY:
            cv2.putText(frame, f"Status: {status_text}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if pitch_display is not None:
                cv2.putText(frame, f"Pitch: {pitch_display:.1f} deg "
                                    f"(down > {DOWN_PITCH_THRESHOLD_DEG})",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 2)
            if ENABLE_PHONE_DETECTION:
                phone_text = "Phone: YES" if phone_recently_seen else "Phone: no"
                cv2.putText(frame, phone_text, (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Lock-In Guardian", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    face_mesh.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
