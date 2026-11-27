from collections import Counter
from pathlib import Path
import time

import cv2
import numpy as np

from vision.pose_detector import (
    DEFAULT_MODEL_PATH,
    HEIGHT,
    PoseDetector,
    WIDTH,
    fit_frame,
)
from vision.rep_counter import RepCounter
from vision.rules import Exercise, RULES, VisionResult

COUNTDOWN_SECONDS = 3.0
CORRECTION_DISPLAY_SECONDS = 2.5


def open_available_camera(preferred_index: int = 0) -> cv2.VideoCapture:
    """Windows(CAP_DSHOW) 및 기본 백엔드를 순회하며 사용 가능한 웹캠을 연결합니다."""
    indices = [preferred_index] + [i for i in range(3) if i != preferred_index]
    backends = [cv2.CAP_DSHOW, cv2.CAP_ANY]

    for idx in indices:
        for backend in backends:
            cap = cv2.VideoCapture(idx, backend)
            if not cap.isOpened():
                cap.release()
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
            for _ in range(10):
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    return cap
                time.sleep(0.05)
            cap.release()

    raise RuntimeError(
        "웹캠을 열 수 없습니다. 카메라 권한 또는 디스코드 등 타 앱의 카메라 점유 여부를 확인해 주세요."
    )


class VisionEngine:
    """
    웹캠 프레임 1장을 입력받아 자세 인식, 카운트다운, 반복 횟수/정확도 판정,
    오버레이 프레임 생성을 일괄 처리하는 컴퓨터 비전 코어 엔진입니다.
    """

    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH, skip_align: bool = False):
        self.detector = PoseDetector(model_path=model_path)
        self.skip_align = skip_align
        self.exercise = Exercise.SQUAT
        self.reset(self.exercise)

    def reset(self, exercise: Exercise | str | None = None):
        if exercise is not None:
            self.exercise = Exercise(exercise)
        self.counter = RepCounter(self.exercise)
        self.phase = "exercise" if self.skip_align else "align"
        self.deadline = 0.0
        self.countdown_frame_at: float | None = None
        self.reps = 0
        self.rejections: Counter[str] = Counter()
        self.feedback = f"{RULES[self.exercise].view_guide} 가이드에 맞춰 자세를 잡아주세요."
        self.visible_correction = None
        self.correction_until = 0.0

    def process_frame(
        self,
        raw_frame_bgr: np.ndarray,
        mirror: bool = True,
        now: float | None = None,
    ) -> tuple[np.ndarray, VisionResult]:
        now = time.monotonic() if now is None else now
        if mirror:
            raw_frame_bgr = cv2.flip(raw_frame_bgr, 1)

        frame = fit_frame(raw_frame_bgr)
        landmarks = self.detector.detect(frame, now)
        observation, aligned, active_side, align_hint = self.detector.evaluate(
            landmarks, self.exercise, is_mirrored=mirror
        )

        countdown_left = 0
        rep_completed = False

        # 1. 정렬 및 3초 카운트다운 단계
        if self.phase in ("align", "countdown"):
            if not aligned:
                self.phase = "align"
                self.feedback = align_hint
                self.countdown_frame_at = None
            elif self.phase == "align" or (
                self.countdown_frame_at is not None and now - self.countdown_frame_at > 0.45
            ):
                self.phase = "countdown"
                self.deadline = now + COUNTDOWN_SECONDS
                self.countdown_frame_at = now
                self.feedback = "위치 확인 완료! 자세를 유지하세요."
            else:
                self.countdown_frame_at = now
                countdown_left = max(0, int(np.ceil(self.deadline - now)))
                if now >= self.deadline:
                    self.phase = "exercise"
                    self.counter.reset()
                    self.feedback = "시작 자세를 잠시 유지하세요."

        # 2. 본 운동 카운팅 및 교정 판정 단계
        elif self.phase == "exercise":
            update = self.counter.update(observation, now)
            self.feedback = update.feedback
            if update.correction:
                self.visible_correction = update.correction
                self.correction_until = now + CORRECTION_DISPLAY_SECONDS

            if update.result is True:
                self.reps += 1
                rep_completed = True
            elif update.result is False and update.correction is not None:
                self.rejections[update.correction.reason] += 1

        # 3. 시각화 오버레이 렌더링
        self.detector.draw_overlay(
            frame=frame,
            landmarks=landmarks,
            exercise=self.exercise,
            aligned=aligned,
            active_side=active_side,
            observation=observation,
            show_guide=(self.phase in ("align", "countdown")),
        )

        hint = (
            self.visible_correction
            if self.phase == "exercise" and now < self.correction_until
            else None
        )
        judged_attempts = self.reps + self.rejections.total()
        accuracy = round((self.reps / judged_attempts) * 100.0, 1) if judged_attempts > 0 else 0.0

        result = VisionResult(
            exercise=self.exercise.value,
            phase=self.phase,
            aligned=aligned,
            countdown_left=countdown_left,
            reps=self.reps,
            judged_attempts=judged_attempts,
            accuracy=accuracy,
            angle=round(observation.angle, 1) if observation else None,
            left_angle=round(observation.left_angle, 1) if observation else 0.0,
            right_angle=round(observation.right_angle, 1) if observation else 0.0,
            feedback=hint.message if hint else self.feedback,
            feedback_detail=hint.detail if hint else "",
            is_correction=bool(hint),
            rep_completed=rep_completed,
            rejections=dict(self.rejections),
        )
        return frame, result

    def close(self):
        self.detector.close()