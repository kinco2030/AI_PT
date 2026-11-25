from collections import deque
import numpy as np
from vision.angle_utils import calculate_angle

SIDE_LANDMARKS = {
    "LEFT": {
        "shoulder": 11,
        "elbow": 13,
        "wrist": 15,
        "hip": 23,
        "knee": 25,
        "ankle": 27,
    },
    "RIGHT": {
        "shoulder": 12,
        "elbow": 14,
        "wrist": 16,
        "hip": 24,
        "knee": 26,
        "ankle": 28,
    },
}


class ExerciseAnalyzer:
    def __init__(self, smoothing_window: int = 4):
        self.exercise = "pushup"
        self.count = 0
        self.stage = "READY"
        self.feedback = "운동을 시작해 주세요."
        self.angle_buffer = deque(maxlen=smoothing_window)
        self.left_angle_buffer = deque(maxlen=smoothing_window)
        self.right_angle_buffer = deque(maxlen=smoothing_window)
        self.rep_accuracies: list[float] = []

    def reset(self, exercise: str | None = None):
        if exercise:
            self.exercise = exercise
        self.count = 0
        self.stage = "READY"
        if self.exercise == "pushup":
            self.feedback = "정면을 바라보고 양팔을 펴서 준비 자세를 취해주세요."
        elif self.exercise == "squat":
            self.feedback = "정면을 바라보고 어깨너비로 서서 준비 자세를 취해주세요."
        else:
            self.feedback = "웹캠이 내 오른쪽에 오도록 측면으로 누워 준비 자세를 취해주세요."
        self.angle_buffer.clear()
        self.left_angle_buffer.clear()
        self.right_angle_buffer.clear()
        self.rep_accuracies.clear()

    def _get_point(
        self,
        landmarks,
        idx: int,
        frame_shape: tuple[int, int] | None = None,
    ) -> tuple[float, float]:
        if frame_shape is not None:
            h, w = frame_shape
            return (landmarks[idx].x * w, landmarks[idx].y * h)
        return (landmarks[idx].x, landmarks[idx].y)

    def _smooth(self, buffer: deque, raw_value: float) -> float:
        buffer.append(raw_value)
        return float(np.mean(buffer))

    def analyze(
        self,
        landmarks,
        active_side: str | None = None,
        frame_shape: tuple[int, int] | None = None,
    ) -> dict:
        left_angle = 0.0
        right_angle = 0.0

        if self.exercise == "pushup":
            raw_left = calculate_angle(
                self._get_point(landmarks, 11, frame_shape),
                self._get_point(landmarks, 13, frame_shape),
                self._get_point(landmarks, 15, frame_shape),
            )
            raw_right = calculate_angle(
                self._get_point(landmarks, 12, frame_shape),
                self._get_point(landmarks, 14, frame_shape),
                self._get_point(landmarks, 16, frame_shape),
            )
            left_angle = self._smooth(self.left_angle_buffer, raw_left)
            right_angle = self._smooth(self.right_angle_buffer, raw_right)
            angle = (left_angle + right_angle) / 2.0
            self._update_pushup_front(angle, abs(left_angle - right_angle))

        elif self.exercise == "squat":
            raw_left = calculate_angle(
                self._get_point(landmarks, 23, frame_shape),
                self._get_point(landmarks, 25, frame_shape),
                self._get_point(landmarks, 27, frame_shape),
            )
            raw_right = calculate_angle(
                self._get_point(landmarks, 24, frame_shape),
                self._get_point(landmarks, 26, frame_shape),
                self._get_point(landmarks, 28, frame_shape),
            )
            left_angle = self._smooth(self.left_angle_buffer, raw_left)
            right_angle = self._smooth(self.right_angle_buffer, raw_right)
            angle = (left_angle + right_angle) / 2.0
            self._update_squat_front(angle, abs(left_angle - right_angle))

        elif self.exercise == "situp":
            # check_posture_alignment가 판별한 카메라 방향 관절 세트 사용
            side_key = active_side if active_side in SIDE_LANDMARKS else "LEFT"
            idx = SIDE_LANDMARKS[side_key]
            raw_hip = calculate_angle(
                self._get_point(landmarks, idx["shoulder"], frame_shape),
                self._get_point(landmarks, idx["hip"], frame_shape),
                self._get_point(landmarks, idx["knee"], frame_shape),
            )
            angle = self._smooth(self.angle_buffer, raw_hip)
            right_angle = angle
            self._update_situp_right(angle)

        else:
            angle = 0.0
            self.feedback = "지원하지 않는 운동 종목입니다."

        avg_accuracy = float(np.mean(self.rep_accuracies)) if self.rep_accuracies else 0.0

        return {
            "exercise": self.exercise,
            "count": self.count,
            "stage": self.stage,
            "angle": round(angle, 1),
            "left_angle": round(left_angle, 1),
            "right_angle": round(right_angle, 1),
            "feedback": self.feedback,
            "accuracy": round(avg_accuracy, 1),
        }

    def _update_pushup_front(self, avg_elbow_angle: float, angle_diff: float):
        if avg_elbow_angle < 135 and angle_diff > 35.0:
            self.feedback = f"좌우 팔 각도 차이({angle_diff:.0f}°)가 큽니다. 양팔 균형을 맞춰주세요!"
            return

        if avg_elbow_angle > 140:
            if self.stage == "DOWN":
                self.count += 1
                self.feedback = f"{self.count}회 성공! 양팔을 균형 있게 폈습니다."
            elif self.stage == "READY":
                self.feedback = "정면 준비 완료. 팔꿈치를 굽혀 내려가세요."
            self.stage = "UP"

        elif avg_elbow_angle < 120:
            if self.stage == "UP":
                depth_penalty = max(0.0, avg_elbow_angle - 110.0)
                symmetry_penalty = angle_diff * 0.3
                score = max(70.0, 100.0 - depth_penalty - symmetry_penalty)
                self.rep_accuracies.append(score)
            self.stage = "DOWN"
            self.feedback = "충분히 내려갔습니다! 바닥을 밀어 올라오세요."

        elif self.stage == "UP" and 120 <= avg_elbow_angle <= 138:
            self.feedback = f"내려가는 중... 팔 각도를 120° 이하로 낮춰보세요. (현재 {avg_elbow_angle:.0f}°)"

    def _update_squat_front(self, avg_knee_angle: float, angle_diff: float):
        if avg_knee_angle < 140 and angle_diff > 30.0:
            self.feedback = f"무릎 좌우 비대칭({angle_diff:.0f}°)이 감지되었습니다. 양발에 균등하게 체중을 실으세요!"
            return

        if avg_knee_angle > 155:
            if self.stage == "DOWN":
                self.count += 1
                self.feedback = f"{self.count}회 성공! 좋은 정면 스쿼트 자세입니다."
            elif self.stage == "READY":
                self.feedback = "정면 준비 완료. 무릎을 굽혀 천천히 앉아주세요."
            self.stage = "UP"

        elif avg_knee_angle < 115:
            if self.stage == "UP":
                depth_penalty = max(0.0, avg_knee_angle - 100.0)
                symmetry_penalty = angle_diff * 0.3
                score = max(70.0, 100.0 - depth_penalty - symmetry_penalty)
                self.rep_accuracies.append(score)
            self.stage = "DOWN"
            self.feedback = "충분히 앉았습니다! 허리를 펴고 일어나세요."

        elif self.stage == "UP" and 115 <= avg_knee_angle <= 145:
            self.feedback = f"앉는 중... 무릎 각도를 115° 이하로 더 낮춰보세요. (현재 {avg_knee_angle:.0f}°)"

    def _update_situp_right(self, hip_angle: float):
        # 무릎을 굽히고 누웠을 때 기본 각도(108도 이상)를 고려해 완화
        if hip_angle > 108:
            self.stage = "DOWN"
            self.feedback = "준비 자세 완료. 복근 힘으로 상체를 일으키세요."
        elif hip_angle < 75 and self.stage == "DOWN":
            self.stage = "UP"
            self.count += 1
            score = max(70.0, 100.0 - max(0.0, hip_angle - 60.0))
            self.rep_accuracies.append(score)
            self.feedback = f"{self.count}회 성공! 천천히 바닥으로 내려가세요."
        elif self.stage == "DOWN" and 75 <= hip_angle <= 105:
            self.feedback = f"올라오는 중... 상체를 조금 더 일으켜 주세요. (현재 {hip_angle:.0f}°)"