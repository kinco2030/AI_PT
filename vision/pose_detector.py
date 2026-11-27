from collections import deque
from pathlib import Path
import tempfile
import urllib.request

import cv2
import mediapipe as mp
import numpy as np

from vision.rules import (
    Exercise,
    Observation,
    PUSH_UP_ASYM,
    RULES,
    SQUAT_ASYM,
    VISIBILITY,
)

WIDTH, HEIGHT = 960, 540
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
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

# 종목별 시작 자세 실루엣 가이드 좌표 (정면 스쿼트 / 정면 팔굽혀펴기 / 오른쪽 측면 윗몸일으키기)
SILHOUETTE_GUIDES = {
    Exercise.SQUAT: {
        "head": (0.50, 0.16),
        "torso": ((0.43, 0.28), (0.57, 0.28), (0.55, 0.52), (0.45, 0.52), (0.43, 0.28)),
        "limbs": (
            ((0.43, 0.28), (0.38, 0.43), (0.36, 0.56)),
            ((0.57, 0.28), (0.62, 0.43), (0.64, 0.56)),
            ((0.45, 0.52), (0.44, 0.71), (0.43, 0.89)),
            ((0.55, 0.52), (0.56, 0.71), (0.57, 0.89)),
        ),
    },
    Exercise.PUSH_UP: {
        "head": (0.50, 0.24),
        "torso": ((0.37, 0.36), (0.63, 0.36), (0.58, 0.58), (0.42, 0.58), (0.37, 0.36)),
        "limbs": (
            ((0.37, 0.36), (0.31, 0.56), (0.32, 0.76)),
            ((0.63, 0.36), (0.69, 0.56), (0.68, 0.76)),
        ),
    },
    Exercise.SIT_UP: {
        "head": (0.76, 0.72),
        "torso": ((0.65, 0.73), (0.48, 0.75)),
        "limbs": (
            ((0.48, 0.75), (0.35, 0.55), (0.24, 0.76)),
            ((0.63, 0.67), (0.55, 0.68)),
        ),
    },
}


def prepare_model(model_path: Path = DEFAULT_MODEL_PATH) -> Path:
    if model_path.is_file() and model_path.stat().st_size > 1_000_000:
        return model_path
    print("[INFO] Google MediaPipe 자세 인식 모델을 내려받고 있습니다...", flush=True)
    with urllib.request.urlopen(MODEL_URL, timeout=60) as response:
        data = response.read()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=model_path.parent, suffix=".download", delete=False) as stream:
        tmp = Path(stream.name)
        stream.write(data)
    try:
        tmp.replace(model_path)
    finally:
        tmp.unlink(missing_ok=True)
    return model_path


def fit_frame(raw: np.ndarray) -> np.ndarray:
    """입력 웹캠 프레임을 16:9(960x540) 캔버스 비율에 맞춰 레터박스 패딩합니다."""
    height, width = raw.shape[:2]
    scale = min(WIDTH / width, HEIGHT / height)
    fitted = cv2.resize(raw, (round(width * scale), round(height * scale)))
    canvas = np.full((HEIGHT, WIDTH, 3), (24, 22, 19), dtype=np.uint8)
    y, x = (HEIGHT - fitted.shape[0]) // 2, (WIDTH - fitted.shape[1]) // 2
    canvas[y : y + fitted.shape[0], x : x + fitted.shape[1]] = fitted
    return canvas


def calculate_angle(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
) -> float:
    a_arr = np.array(a, dtype=np.float64)
    b_arr = np.array(b, dtype=np.float64)
    c_arr = np.array(c, dtype=np.float64)
    radians = np.arctan2(c_arr[1] - b_arr[1], c_arr[0] - b_arr[0]) - np.arctan2(
        a_arr[1] - b_arr[1], a_arr[0] - b_arr[0]
    )
    angle = np.abs(radians * 180.0 / np.pi)
    return float(360.0 - angle if angle > 180.0 else angle)


def _draw_angle_arc(
    img: np.ndarray,
    pt_a: tuple[int, int],
    pt_b: tuple[int, int],
    pt_c: tuple[int, int],
    angle_deg: float,
    target_threshold: float,
    label: str,
    offset_x: int = 14,
):
    color = (117, 238, 146) if angle_deg <= target_threshold else (0, 225, 255)
    cv2.line(img, pt_a, pt_b, color, 4, cv2.LINE_AA)
    cv2.line(img, pt_b, pt_c, color, 4, cv2.LINE_AA)

    ang1 = np.degrees(np.arctan2(pt_a[1] - pt_b[1], pt_a[0] - pt_b[0]))
    ang2 = np.degrees(np.arctan2(pt_c[1] - pt_b[1], pt_c[0] - pt_b[0]))
    diff = (ang2 - ang1) % 360
    start_ang, end_ang = (ang1, ang1 + diff) if diff <= 180 else (ang2, ang2 + (360 - diff))
    cv2.ellipse(img, pt_b, (28, 28), 0, start_ang, end_ang, color, 2, cv2.LINE_AA)

    pos = (max(10, min(WIDTH - 135, pt_b[0] + offset_x)), max(30, pt_b[1] - 10))
    text = f"{label}:{angle_deg:.0f} deg"
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 4, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)


class PoseDetector:
    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH, smoothing_window: int = 4):
        resolved = prepare_model(model_path)
        self.landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(
            mp.tasks.vision.PoseLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_buffer=resolved.read_bytes()),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.45,
                min_pose_presence_confidence=0.45,
                min_tracking_confidence=0.45,
            )
        )
        self.previous_timestamp = -1
        self._current_exercise: Exercise | None = None
        self._angle_buf = deque(maxlen=smoothing_window)
        self._left_buf = deque(maxlen=smoothing_window)
        self._right_buf = deque(maxlen=smoothing_window)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.landmarker.close()

    def close(self):
        self.landmarker.close()

    def _reset_buffers(self, exercise: Exercise):
        if self._current_exercise != exercise:
            self._current_exercise = exercise
            self._angle_buf.clear()
            self._left_buf.clear()
            self._right_buf.clear()

    @staticmethod
    def _vis(landmarks, idx: int) -> float:
        v = landmarks[idx].visibility
        return float(v) if v is not None else 1.0

    @staticmethod
    def _pt(landmarks, idx: int) -> tuple[float, float]:
        return (landmarks[idx].x * WIDTH, landmarks[idx].y * HEIGHT)

    @staticmethod
    def _smooth(buf: deque, val: float) -> float:
        buf.append(val)
        return float(np.mean(buf))

    @staticmethod
    def _front_squat_knee_angle(
        hip: tuple[float, float],
        knee: tuple[float, float],
        ankle: tuple[float, float],
    ) -> float:
        """
        정면 웹캠에서는 무릎이 카메라 앞(깊이 방향)으로 접혀 2D 평면 각도 변화가 작으므로,
        정강이 길이 대비 엉덩이 수직 하강 비율을 각도로 환산해 함께 반영합니다.
        """
        raw_2d = calculate_angle(hip, knee, ankle)
        shin_len = float(np.hypot(ankle[0] - knee[0], ankle[1] - knee[1]))
        if shin_len < 15.0:
            return raw_2d

        # 기립 시(thigh_drop >= 0.85 * shin_len) -> 170도, 허벅지 수평(thigh_drop == 0) -> 90도
        thigh_drop = knee[1] - hip[1]
        ratio = float(np.clip(thigh_drop / (shin_len * 0.85), -0.2, 1.0))
        projected_angle = 90.0 + 80.0 * ratio
        return min(raw_2d, projected_angle)

    def detect(self, frame_bgr: np.ndarray, now: float):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp = max(self.previous_timestamp + 1, int(now * 1000))
        self.previous_timestamp = timestamp
        response = self.landmarker.detect_for_video(image, timestamp)
        if not response.pose_landmarks:
            return None
        return response.pose_landmarks[0]

    def evaluate(
        self,
        landmarks,
        exercise: Exercise,
        is_mirrored: bool = True,
    ) -> tuple[Observation | None, bool, str | None, str]:
        """
        랜드마크와 운동 종목을 받아 (관측각도, 정렬여부, 주시측면, 정렬안내문구)를 반환합니다.
        """
        self._reset_buffers(exercise)
        if not landmarks:
            return None, False, None, "인식된 사람이 없습니다. 카메라 앞에 위치해 주세요."

        # 1. 팔굽혀펴기: 웹캠 정면(FRONT) - 양쪽 어깨·팔꿈치·손목 각도
        if exercise == Exercise.PUSH_UP:
            if min(self._vis(landmarks, i) for i in (11, 12, 13, 14, 15, 16)) < VISIBILITY:
                return None, False, None, "양쪽 어깨, 팔꿈치, 손목이 모두 보이도록 화면 안으로 들어와 주세요."

            shoulder_dx = abs(landmarks[11].x - landmarks[12].x)
            shoulder_dy = abs(landmarks[11].y - landmarks[12].y)
            if shoulder_dx < 0.14:
                return None, False, "FRONT", "웹캠을 정면으로 바라보고 양어깨가 넓게 보이도록 자세를 잡아주세요."
            if shoulder_dy > 0.15:
                return None, False, "FRONT", "양쪽 어깨의 높이를 수평으로 맞춰 주세요."

            raw_l = calculate_angle(self._pt(landmarks, 11), self._pt(landmarks, 13), self._pt(landmarks, 15))
            raw_r = calculate_angle(self._pt(landmarks, 12), self._pt(landmarks, 14), self._pt(landmarks, 16))
            left_ang = self._smooth(self._left_buf, raw_l)
            right_ang = self._smooth(self._right_buf, raw_r)
            avg_ang = (left_ang + right_ang) / 2.0
            fault = PUSH_UP_ASYM if (avg_ang < 135.0 and abs(left_ang - right_ang) > 35.0) else None
            return Observation(avg_ang, left_ang, right_ang, fault), True, "FRONT", "정면 정렬 완료"

        # 2. 스쿼트: 웹캠 정면(FRONT) - 원거리 전신 가시성 완화 + 정면 깊이 투영 각도 적용
        if exercise == Exercise.SQUAT:
            core_visible = min(self._vis(landmarks, i) for i in (11, 12, 23, 24, 25, 26)) >= 0.40
            ankles_visible = min(self._vis(landmarks, i) for i in (27, 28)) >= 0.25
            ankles_in_frame = max(landmarks[27].y, landmarks[28].y) <= 0.99

            if not (core_visible and ankles_visible and ankles_in_frame):
                return None, False, None, "머리부터 발목까지 전신이 화면에 들어오도록 뒤로 물러서 주세요."

            shoulder_dx = abs(landmarks[11].x - landmarks[12].x)
            hip_dx = abs(landmarks[23].x - landmarks[24].x)
            if shoulder_dx < 0.10 or hip_dx < 0.045:
                return None, False, "FRONT", "웹캠을 정면으로 바라보고 양발을 어깨너비로 벌려 서주세요."

            raw_l = self._front_squat_knee_angle(
                self._pt(landmarks, 23), self._pt(landmarks, 25), self._pt(landmarks, 27)
            )
            raw_r = self._front_squat_knee_angle(
                self._pt(landmarks, 24), self._pt(landmarks, 26), self._pt(landmarks, 28)
            )
            left_ang = self._smooth(self._left_buf, raw_l)
            right_ang = self._smooth(self._right_buf, raw_r)
            avg_ang = (left_ang + right_ang) / 2.0
            fault = SQUAT_ASYM if (avg_ang < 140.0 and abs(left_ang - right_ang) > 30.0) else None
            return Observation(avg_ang, left_ang, right_ang, fault), True, "FRONT", "정면 정렬 완료"

        # 3. 윗몸일으키기: 웹캠 오른쪽 측면(RIGHT-SIDE) - 어깨·엉덩이·무릎 각도
        left_vis = float(np.mean([self._vis(landmarks, i) for i in (11, 23, 25)]))
        right_vis = float(np.mean([self._vis(landmarks, i) for i in (12, 24, 26)]))
        if max(left_vis, right_vis) < VISIBILITY:
            return None, False, None, "어깨, 엉덩이, 무릎이 모두 보이도록 카메라와 거리를 조절해 주세요."

        active_side = "LEFT" if left_vis >= right_vis else "RIGHT"
        s_idx, h_idx, k_idx = (11, 23, 25) if active_side == "LEFT" else (12, 24, 26)

        hip_x = landmarks[h_idx].x
        knee_x = landmarks[k_idx].x
        is_webcam_on_my_right = (knee_x < hip_x) if is_mirrored else (knee_x > hip_x)
        if not is_webcam_on_my_right:
            return None, False, active_side, "웹캠이 내 오른쪽에 오도록 반대 방향으로 누워주세요."

        body_dx = abs(landmarks[s_idx].x - hip_x)
        shoulder_overlap_dx = abs(landmarks[11].x - landmarks[12].x)
        if body_dx < 0.08 and shoulder_overlap_dx > 0.25:
            return None, False, active_side, "오른쪽 옆모습이 보이도록 웹캠과 수직 방향으로 누워주세요."

        raw_hip = calculate_angle(self._pt(landmarks, s_idx), self._pt(landmarks, h_idx), self._pt(landmarks, k_idx))
        hip_ang = self._smooth(self._angle_buf, raw_hip)
        return Observation(hip_ang, 0.0, hip_ang, None), True, active_side, "오른쪽 측면 정렬 완료"

    def draw_guide(self, frame: np.ndarray, exercise: Exercise, aligned: bool):
        """프로토타입 스타일의 반투명 인체 실루엣 가이드와 코너 브래킷을 렌더링합니다."""
        color = (146, 238, 117) if aligned else (216, 184, 139)
        spec = SILHOUETTE_GUIDES[exercise]

        head_pt = tuple(np.round(np.array(spec["head"]) * (WIDTH, HEIGHT)).astype(np.int32))
        torso_pts = np.round(np.array(spec["torso"]) * (WIDTH, HEIGHT)).astype(np.int32)
        limb_lines = [
            np.round(np.array(limb) * (WIDTH, HEIGHT)).astype(np.int32) for limb in spec["limbs"]
        ]

        faded = frame.copy()
        cv2.circle(faded, head_pt, 26, color, -1, cv2.LINE_AA)
        if len(torso_pts) > 2:
            cv2.fillPoly(faded, [torso_pts], color, cv2.LINE_AA)
        cv2.polylines(faded, [torso_pts, *limb_lines], False, color, 26, cv2.LINE_AA)
        cv2.addWeighted(faded, 0.14, frame, 0.86, 0, dst=frame)

        cv2.circle(frame, head_pt, 24, color, 2, cv2.LINE_AA)
        cv2.polylines(frame, [torso_pts, *limb_lines], False, color, 2, cv2.LINE_AA)

        # 코너 브래킷
        for x, dx in ((35, 1), (WIDTH - 35, -1)):
            for y, dy in ((30, 1), (HEIGHT - 30, -1)):
                cv2.line(frame, (x, y), (x + 28 * dx, y), color, 2, cv2.LINE_AA)
                cv2.line(frame, (x, y), (x, y + 28 * dy), color, 2, cv2.LINE_AA)

    def draw_overlay(
        self,
        frame: np.ndarray,
        landmarks,
        exercise: Exercise,
        aligned: bool,
        active_side: str | None,
        observation: Observation | None,
        show_guide: bool = True,
    ):
        if show_guide:
            self.draw_guide(frame, exercise, aligned)

        if not landmarks:
            return

        points = {}
        for idx, lm in enumerate(landmarks):
            min_vis = 0.25 if idx in (27, 28, 29, 30, 31, 32) else 0.4
            if self._vis(landmarks, idx) >= min_vis:
                points[idx] = (int(lm.x * WIDTH), int(lm.y * HEIGHT))

        for s_idx, e_idx in POSE_CONNECTIONS:
            if s_idx in points and e_idx in points:
                cv2.line(frame, points[s_idx], points[e_idx], (143, 235, 131), 2, cv2.LINE_AA)

        target_ang = RULES[exercise].target
        if observation is not None:
            if exercise == Exercise.PUSH_UP:
                if all(k in points for k in (11, 13, 15)) and observation.left_angle > 0:
                    _draw_angle_arc(frame, points[11], points[13], points[15], observation.left_angle, target_ang, "L", 14)
                if all(k in points for k in (12, 14, 16)) and observation.right_angle > 0:
                    _draw_angle_arc(frame, points[12], points[14], points[16], observation.right_angle, target_ang, "R", -125)
            elif exercise == Exercise.SQUAT:
                if all(k in points for k in (23, 25, 27)) and observation.left_angle > 0:
                    _draw_angle_arc(frame, points[23], points[25], points[27], observation.left_angle, target_ang, "L", 14)
                if all(k in points for k in (24, 26, 28)) and observation.right_angle > 0:
                    _draw_angle_arc(frame, points[24], points[26], points[28], observation.right_angle, target_ang, "R", -125)
            elif exercise == Exercise.SIT_UP:
                triplet = (11, 23, 25) if active_side == "LEFT" else (12, 24, 26)
                if all(k in points for k in triplet) and observation.angle > 0:
                    _draw_angle_arc(frame, points[triplet[0]], points[triplet[1]], points[triplet[2]], observation.angle, target_ang, "HIP", 14)

        for pt in points.values():
            cv2.circle(frame, pt, 4, (235, 255, 230), -1, cv2.LINE_AA)