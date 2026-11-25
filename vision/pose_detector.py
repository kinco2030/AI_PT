import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from vision.angle_utils import check_posture_alignment

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "pose_landmarker_lite.task"

POSE_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12), (11, 23), (12, 24), (23, 24),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
)


def _ensure_model_file(model_path: Path) -> str:
    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] PoseLandmarker 모델 다운로드 중... -> {model_path}")
        urllib.request.urlretrieve(MODEL_URL, str(model_path))
        print("[INFO] 모델 다운로드 완료.")
    return str(model_path)


def _draw_joint_angle_arc(
    img: np.ndarray,
    pt_a: tuple[int, int],
    pt_b: tuple[int, int],
    pt_c: tuple[int, int],
    angle_deg: float,
    target_threshold: float,
    label_prefix: str,
    text_offset_x: int = 12,
):
    arc_color = (46, 204, 113) if angle_deg < target_threshold else (0, 230, 255)

    cv2.line(img, pt_a, pt_b, arc_color, 4)
    cv2.line(img, pt_b, pt_c, arc_color, 4)

    ang1 = np.degrees(np.arctan2(pt_a[1] - pt_b[1], pt_a[0] - pt_b[0]))
    ang2 = np.degrees(np.arctan2(pt_c[1] - pt_b[1], pt_c[0] - pt_b[0]))
    diff = (ang2 - ang1) % 360

    if diff <= 180:
        start_ang, end_ang = ang1, ang1 + diff
    else:
        start_ang, end_ang = ang2, ang2 + (360 - diff)

    cv2.ellipse(img, pt_b, (28, 28), 0, start_ang, end_ang, arc_color, 2)

    text_pos = (pt_b[0] + text_offset_x, pt_b[1] - 10)
    cv2.putText(
        img,
        f"{label_prefix}:{angle_deg:.0f} deg",
        text_pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 0),
        4,
    )
    cv2.putText(
        img,
        f"{label_prefix}:{angle_deg:.0f} deg",
        text_pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        arc_color,
        2,
    )


class PoseDetector:
    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        min_detection_confidence: float = 0.45,
        min_tracking_confidence: float = 0.45,
    ):
        resolved_model_path = _ensure_model_file(model_path)

        base_options = python.BaseOptions(model_asset_path=resolved_model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        self._last_timestamp_ms = 0

    def extract_landmarks(
        self,
        frame_bgr: np.ndarray,
        exercise_name: str = "pushup",
        is_mirrored: bool = True,
    ):
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        now_ms = int(time.monotonic() * 1000)
        timestamp_ms = max(self._last_timestamp_ms + 1, now_ms)
        self._last_timestamp_ms = timestamp_ms

        results = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        if not results.pose_landmarks:
            return None, None, False, "인식된 사람이 없습니다. 카메라 앞에 위치해 주세요."

        landmarks = results.pose_landmarks[0]
        active_side, is_aligned, guide_msg = check_posture_alignment(
            landmarks, exercise_name, is_mirrored=is_mirrored
        )
        return landmarks, active_side, is_aligned, guide_msg

    def draw_overlay(
        self,
        frame_bgr: np.ndarray,
        landmarks,
        is_aligned: bool,
        exercise_name: str,
        count: int,
        stage: str,
        angle: float,
        left_angle: float = 0.0,
        right_angle: float = 0.0,
        active_side: str | None = None,
    ) -> np.ndarray:
        annotated = frame_bgr.copy()
        h, w, _ = annotated.shape
        is_front_mode = exercise_name in ("pushup", "squat")

        # 1. 종목별 가이드라인
        guide_color = (46, 204, 113) if is_aligned else (0, 140, 255)
        if is_front_mode:
            margin_x, margin_y = int(w * 0.08), int(h * 0.08)
            cv2.rectangle(annotated, (margin_x, margin_y), (w - margin_x, h - margin_y), guide_color, 2)
            cv2.line(annotated, (margin_x, h // 2), (w - margin_x, h // 2), (200, 200, 200), 1)
        else:
            # 싯업: 누운 자세에 맞춘 가로형 가이드 박스
            margin_x, margin_y = int(w * 0.06), int(h * 0.14)
            cv2.rectangle(annotated, (margin_x, margin_y), (w - margin_x, h - margin_y), guide_color, 2)
            cv2.line(annotated, (w // 2, margin_y), (w // 2, h - margin_y), (200, 200, 200), 1)

        # 2. 전체 관절 스켈레톤 및 종목별 핵심 관절 각도 호 렌더링
        if landmarks:
            points = {}
            for idx, lm in enumerate(landmarks):
                vis = lm.visibility if lm.visibility is not None else 1.0
                if vis >= 0.4:
                    points[idx] = (int(lm.x * w), int(lm.y * h))

            for start_idx, end_idx in POSE_CONNECTIONS:
                if start_idx in points and end_idx in points:
                    cv2.line(annotated, points[start_idx], points[end_idx], guide_color, 2)

            if exercise_name == "pushup":
                if all(k in points for k in (11, 13, 15)) and left_angle > 0:
                    _draw_joint_angle_arc(
                        annotated, points[11], points[13], points[15], left_angle, 120.0, "L", text_offset_x=14
                    )
                if all(k in points for k in (12, 14, 16)) and right_angle > 0:
                    _draw_joint_angle_arc(
                        annotated, points[12], points[14], points[16], right_angle, 120.0, "R", text_offset_x=-115
                    )

            elif exercise_name == "squat":
                if all(k in points for k in (23, 25, 27)) and left_angle > 0:
                    _draw_joint_angle_arc(
                        annotated, points[23], points[25], points[27], left_angle, 115.0, "L", text_offset_x=14
                    )
                if all(k in points for k in (24, 26, 28)) and right_angle > 0:
                    _draw_joint_angle_arc(
                        annotated, points[24], points[26], points[28], right_angle, 115.0, "R", text_offset_x=-115
                    )

            elif exercise_name == "situp":
                hip_triplet = (11, 23, 25) if active_side == "LEFT" else (12, 24, 26)
                if not all(k in points for k in hip_triplet):
                    hip_triplet = (11, 23, 25) if all(k in points for k in (11, 23, 25)) else (12, 24, 26)

                if all(k in points for k in hip_triplet) and angle > 0:
                    _draw_joint_angle_arc(
                        annotated,
                        points[hip_triplet[0]],
                        points[hip_triplet[1]],
                        points[hip_triplet[2]],
                        angle,
                        75.0,
                        "HIP",
                        text_offset_x=14,
                    )

            for pt in points.values():
                cv2.circle(annotated, pt, 4, (255, 255, 255), -1)
                cv2.circle(annotated, pt, 2, guide_color, -1)

        # 3. 상단 상태 HUD 박스
        view_label = "FRONT" if is_front_mode else "RIGHT-SIDE"
        hud_height = 142 if is_front_mode else 115
        cv2.rectangle(annotated, (15, 15), (425, hud_height), (25, 25, 25), -1)
        cv2.putText(
            annotated,
            f"EXERCISE: {exercise_name.upper()} ({view_label})",
            (25, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            annotated,
            f"COUNT: {count}  |  STAGE: {stage}",
            (25, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (46, 204, 113),
            2,
        )

        if exercise_name == "pushup":
            cv2.putText(
                annotated,
                f"AVG ARM ANGLE: {angle:.1f} deg",
                (25, 102),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (241, 196, 15),
                2,
            )
            cv2.putText(
                annotated,
                f"L-ARM: {left_angle:.1f} deg  |  R-ARM: {right_angle:.1f} deg",
                (25, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 230, 255),
                2,
            )
        elif exercise_name == "squat":
            cv2.putText(
                annotated,
                f"AVG KNEE ANGLE: {angle:.1f} deg",
                (25, 102),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (241, 196, 15),
                2,
            )
            cv2.putText(
                annotated,
                f"L-KNEE: {left_angle:.1f} deg  |  R-KNEE: {right_angle:.1f} deg",
                (25, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 230, 255),
                2,
            )
        else:
            cv2.putText(
                annotated,
                f"HIP ANGLE: {angle:.1f} deg",
                (25, 102),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (241, 196, 15),
                2,
            )

        return annotated

    def close(self):
        self.landmarker.close()