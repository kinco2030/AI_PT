"""AI PT 트레이너 — Windows에서 실행하는 로컬 Gradio 앱.

실행: 저장소 루트에서 uv run --locked python app.py
Python 3.12와 라이브러리는 pyproject.toml, .python-version, uv.lock으로 관리합니다.
모델과 운동 기록은 기본적으로 ai_pt_trainer_files 폴더에 저장됩니다.
AI_PT_TRAINER_HOME 환경 변수로 저장 위치를 변경할 수 있습니다.
"""

import argparse
import asyncio
from collections import Counter
from contextlib import asynccontextmanager
import csv
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime
from enum import StrEnum
import hashlib
from html import escape
import json
import logging
import math
import os
from pathlib import Path
from queue import Empty, Full, Queue
import socket
import sys
import tempfile
from threading import Event, Lock, RLock, Thread, Timer
import time
from time import monotonic
import urllib.request
from uuid import uuid4
import webbrowser


# 최초 실행과 저장 위치

SCRIPT = Path(__file__).resolve()
APP_HOME = Path(os.environ.get(
    "AI_PT_TRAINER_HOME", str(SCRIPT.with_name("ai_pt_trainer_files")),
)).expanduser().resolve()
REPOSITORY_MODEL = SCRIPT.parent / "models" / "pose_landmarker_lite.task"
MODEL_PATH = APP_HOME / "models" / "pose_landmarker_lite.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
MODEL_SHA256 = "59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a"


def prepare_model():
    if MODEL_PATH.is_file() and hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest() == MODEL_SHA256:
        return

    data = REPOSITORY_MODEL.read_bytes() if REPOSITORY_MODEL.is_file() else b""
    if hashlib.sha256(data).hexdigest() != MODEL_SHA256:
        print("Google MediaPipe 자세 모델을 내려받고 있습니다.", flush=True)
        with urllib.request.urlopen(MODEL_URL, timeout=60) as response:
            data = response.read()
    if hashlib.sha256(data).hexdigest() != MODEL_SHA256:
        raise ValueError("내려받은 자세 모델의 체크섬이 예상과 다릅니다.")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=MODEL_PATH.parent, suffix=".download", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(data)
    try:
        temporary.replace(MODEL_PATH)
    finally:
        temporary.unlink(missing_ok=True)


def parse_options():
    parser = argparse.ArgumentParser(description="AI PT 트레이너")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV 카메라 번호")
    parser.add_argument("--port", type=int, default=0, help="로컬 웹 포트 (생략 시 자동 선택)")
    parser.add_argument("--no-browser", action="store_true", help="브라우저 자동 열기 끄기")
    options = parser.parse_args()
    if sys.version_info[:2] != (3, 12):
        parser.error("Python 3.12가 필요합니다. uv run --locked python app.py로 실행하세요.")
    if not 0 <= options.port <= 65535:
        parser.error("포트 번호를 0~65535 사이에서 입력하세요.")
    if options.camera < 0:
        parser.error("카메라 번호를 0 이상으로 입력하세요.")
    return options


os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("GRADIO_TEMP_DIR", str(APP_HOME / ".cache" / "gradio"))
os.environ.setdefault("MPLCONFIGDIR", str(APP_HOME / ".cache" / "matplotlib"))

import cv2
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
import gradio as gr
from matplotlib import font_manager
from matplotlib.figure import Figure
import mediapipe as mp
import numpy as np
import pandas as pd
from starlette.middleware.trustedhost import TrustedHostMiddleware
import uvicorn

LOG = logging.getLogger(__name__)


# 관절 각도와 반복 판정

class Exercise(StrEnum):
    SQUAT = "스쿼트"
    PUSH_UP = "팔굽혀펴기"
    SIT_UP = "윗몸일으키기"


@dataclass(frozen=True)
class Rule:
    start: float
    departure: float
    target: float
    joint: str
    cue: str
    at_target: str
    shallow: str
    count_at_target: bool = False


RULES = {
    Exercise.SQUAT: Rule(
        160, 145, 90, "무릎", "무릎을 굽혀 아래로 내려가세요.",
        "천천히 일어서세요.", "무릎을 조금 더 굽혀 보세요.",
    ),
    Exercise.PUSH_UP: Rule(
        160, 145, 90, "팔꿈치", "팔꿈치를 굽혀 몸을 낮추세요.",
        "몸을 밀어 올려 시작 자세로 돌아오세요.", "팔꿈치를 조금 더 굽혀 보세요.",
    ),
    Exercise.SIT_UP: Rule(
        30, 42, 70, "상체", "반동 없이 상체를 올려 보세요.",
        "상체를 천천히 내려 시작 자세로 돌아오세요.", "상체를 조금 더 올려 보세요.",
        count_at_target=True,
    ),
}

VISIBILITY = 0.58
MAX_FRAME_GAP = 0.38
POSITION_HOLD = 0.12
FAULT_HOLD = 0.20


@dataclass(frozen=True)
class Correction:
    reason: str
    message: str
    detail: str = ""


PUSH_UP_SAG = Correction("허리 처짐", "몸통을 곧게 유지하세요", "어깨부터 발목까지 일직선에 가깝게 맞춰 보세요.")
SQUAT_KNEE = Correction("화면상 무릎이 크게 앞섬", "무릎 위치를 확인하세요", "화면에서 무릎이 발끝보다 크게 앞서 있어요.")
TRACKING_LOST = Correction("관절 추적 끊김", "전신이 화면에 보이게 움직이세요", "관절을 다시 인식할 때까지 횟수 집계를 기다립니다.")
RANGE_SHORT = "동작 범위 부족"
TARGET_HOLD_SHORT = "목표 각도 유지 부족"


def exercise_criteria(exercise: Exercise) -> str:
    rule = RULES[exercise]
    if rule.count_at_target:
        return f"상체 {rule.start:g}° 이하에서 {rule.target:g}° 이상까지 올라오면 인정합니다."
    return f"{rule.joint} 각도 {rule.target:g}° 이하까지 내린 뒤 {rule.start:g}° 이상으로 돌아오면 인정합니다."


def correction_advice(exercise: Exercise, reason: str) -> str:
    if reason == RANGE_SHORT:
        return RULES[exercise].shallow
    if reason == TARGET_HOLD_SHORT:
        return "목표 각도에 도달하면 잠깐 유지한 뒤 돌아오세요."
    for hint in (PUSH_UP_SAG, SQUAT_KNEE):
        if hint.reason == reason:
            return hint.detail
    return ""


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    first, second = a - b, c - b
    scale = np.linalg.norm(first) * np.linalg.norm(second)
    if scale < 1e-8:
        return math.nan
    return math.degrees(math.acos(float(np.clip(np.dot(first, second) / scale, -1, 1))))


@dataclass(frozen=True)
class Observation:
    angle: float
    fault: Correction | None = None


@dataclass(frozen=True)
class RepUpdate:
    result: bool | None
    feedback: str
    correction: Correction | None = None
    interrupted: bool = False


class RepCounter:
    def __init__(self, exercise: Exercise):
        self.rule = RULES[exercise]
        self.reset()

    def reset(self):
        self.phase = "seek_start"
        self.last_seen = None
        self.zone_since = None
        self.zone = None
        self.fault_since = None
        self.fault_seen = None
        self.rep_fault = None
        self.best_angle = None

    @property
    def in_progress(self) -> bool:
        return self.phase in ("descending", "returning")

    def _held(self, condition: bool, now: float) -> bool:
        if not condition:
            self.zone_since = None
            return False
        if self.zone_since is None:
            self.zone_since = now
        return now - self.zone_since >= POSITION_HOLD

    def _move_to(self, phase: str):
        self.phase = phase
        self.zone_since = None
        self.zone = None
        if phase != "returning":
            self.fault_since = None
            self.fault_seen = None
        if phase in ("ready", "descending"):
            self.rep_fault = None
            self.best_angle = None

    def update(self, observation: Observation | None, now: float) -> RepUpdate:
        interrupted = False
        if self.last_seen is not None and now - self.last_seen > MAX_FRAME_GAP:
            interrupted = self.in_progress
            self.reset()
        result = self._follow(observation, now)
        return replace(result, interrupted=True) if interrupted else result

    def _range_correction(self) -> Correction:
        rule = self.rule
        limit = "이상" if rule.count_at_target else "이하"
        reached = self.best_angle >= rule.target if rule.count_at_target else self.best_angle <= rule.target
        if reached:
            return Correction(TARGET_HOLD_SHORT, "목표 자세를 잠깐 유지하세요",
                              f"{rule.target:g}° {limit} 구간을 약 {POSITION_HOLD:g}초 유지해야 합니다.")
        direction = "최고" if rule.count_at_target else "최저"
        return Correction(RANGE_SHORT, rule.shallow,
                          f"이번 {direction} {self.best_angle:.1f}° · 목표 {rule.target:g}° {limit}")

    def _follow(self, observation: Observation | None, now: float) -> RepUpdate:
        if observation is None or not math.isfinite(observation.angle):
            self.zone_since = self.fault_since = None
            self.fault_seen = None
            return RepUpdate(None, TRACKING_LOST.message, TRACKING_LOST)

        self.last_seen = now
        rule = self.rule
        angle = observation.angle
        rising = rule.count_at_target
        at_start = angle <= rule.start if rising else angle >= rule.start
        departed = angle >= rule.departure if rising else angle <= rule.departure
        at_target = angle >= rule.target if rising else angle <= rule.target

        if self.phase == "seek_start":
            if self._held(at_start, now):
                self._move_to("ready")
                return RepUpdate(None, rule.cue)
            return RepUpdate(None, "시작 자세를 잠시 유지하세요.")

        if self.phase == "ready":
            if not departed:
                return RepUpdate(None, rule.cue)
            self._move_to("descending")

        if self.phase == "rearm":
            if self._held(at_start, now):
                self._move_to("ready")
                return RepUpdate(None, rule.cue)
            return RepUpdate(None, rule.at_target)

        correction = None
        if observation.fault != self.fault_seen:
            self.fault_seen = observation.fault
            self.fault_since = now
        if observation.fault and now - self.fault_since >= FAULT_HOLD:
            correction = observation.fault
            self.rep_fault = self.rep_fault or correction

        if self.phase == "descending":
            if self.best_angle is None:
                self.best_angle = angle
            self.best_angle = max(self.best_angle, angle) if rising else min(self.best_angle, angle)
            # 시작 구간과 목표 구간은 겹치지 않는다. 두 구간의 유지 시간은 따로 센다.
            zone = "target" if at_target else "start" if at_start else None
            if zone != self.zone:
                self.zone_since = None
                self.zone = zone
            held = self._held(zone is not None, now)
            if held and zone == "target":
                if rising:
                    fault = self.rep_fault
                    self._move_to("rearm")
                    return RepUpdate(fault is None, rule.at_target, fault)
                self._move_to("returning")
            elif held and zone == "start":
                correction = self._range_correction()
                self._move_to("ready")
                return RepUpdate(False, rule.cue, correction)
        elif self._held(at_start, now):
            fault = self.rep_fault
            self._move_to("ready")
            return RepUpdate(fault is None, rule.cue, fault)

        cue = rule.at_target if self.phase == "returning" else rule.cue
        return RepUpdate(None, cue, correction)


# 루틴, 세트와 휴식

COUNTDOWN_SECONDS = 3
CORRECTION_DISPLAY_SECONDS = 2.5
ACTIVE_PHASES = {"align", "countdown", "exercise", "rest"}


@dataclass(frozen=True)
class Routine:
    exercises: tuple[Exercise, ...]
    reps: int
    sets: int
    rest: int

    def __post_init__(self):
        if not self.exercises or len(set(self.exercises)) != len(self.exercises):
            raise ValueError("서로 다른 운동을 하나 이상 선택하세요.")
        for label, value, low, high in (
            ("횟수", self.reps, 1, 99), ("세트", self.sets, 1, 20),
            ("휴식", self.rest, 0, 300),
        ):
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{label}은 {low}~{high} 사이의 정수로 입력하세요.")


@dataclass(frozen=True)
class Record:
    session_id: str
    finished_at: str
    exercise: str
    target_reps: int
    successful_reps: int
    judged_attempts: int
    active_seconds: float
    sets_completed: int
    sets_target: int
    rejection_reasons: str = ""
    tracking_interruptions: int = 0
    unfinished_attempts: int = 0


@dataclass(frozen=True)
class SetReport:
    exercise: Exercise
    number: int
    successful_reps: int
    reasons: tuple[tuple[str, int], ...]
    tracking_interruptions: int
    unfinished_attempts: int

    @property
    def judged_attempts(self):
        return self.successful_reps + sum(count for _, count in self.reasons)


class Workout:
    def __init__(self, routine: Routine):
        self.routine = routine
        self.id = uuid4().hex
        self.exercise_index = 0
        self.records: list[Record] = []
        self.feedback = ""
        self.correction = None
        self.visible_correction = None
        self.correction_until = 0.0
        self.last_set_report = None
        self.deadline = 0.0
        self.countdown_frame_at = None
        self.phase = "align"
        self._reset_exercise()

    @property
    def exercise(self):
        return self.routine.exercises[self.exercise_index]

    def _reset_exercise(self):
        self.counter = RepCounter(self.exercise)
        self.sets_completed = self.reps = 0
        self.set_reports: list[SetReport] = []
        self.rejections: Counter[str] = Counter()
        self.interruptions = 0
        self.seconds = 0.0
        self.set_started = None
        self.angle = None

    def _start_set(self, now: float):
        self.phase = "exercise"
        self.set_started = now
        self.reps = 0
        self.rejections.clear()
        self.interruptions = 0
        self.visible_correction = None
        self.counter.reset()
        self.feedback = "시작 자세를 잠시 유지하세요."

    def _end_set(self, now: float, completed: bool):
        if self.set_started is None:
            return
        self.seconds += now - self.set_started
        report = SetReport(
            self.exercise, self.sets_completed + 1, self.reps, tuple(self.rejections.most_common()),
            self.interruptions, int(not completed and self.counter.in_progress),
        )
        self.set_reports.append(report)
        if completed:
            self.sets_completed += 1
            self.last_set_report = report
        self.set_started = None

    def _record(self, finished_at: datetime):
        if not self.set_reports:
            return
        reasons: Counter[str] = Counter()
        for report in self.set_reports:
            reasons.update(dict(report.reasons))
        successes = sum(report.successful_reps for report in self.set_reports)
        self.records.append(Record(
            self.id, finished_at.isoformat(timespec="seconds"), self.exercise.value,
            self.routine.reps * self.routine.sets, successes, successes + reasons.total(),
            round(self.seconds, 2), self.sets_completed, self.routine.sets,
            json.dumps(dict(reasons), ensure_ascii=False),
            sum(report.tracking_interruptions for report in self.set_reports),
            sum(report.unfinished_attempts for report in self.set_reports),
        ))

    def stop(self, now: float, finished_at: datetime):
        if self.phase in ACTIVE_PHASES:
            self._end_set(now, completed=False)
            if self.sets_completed < self.routine.sets:
                self._record(finished_at)
            self.phase = "stopped"
            self.angle = None
            self.feedback = "운동을 종료했어요."

    def advance(self, aligned: bool, observation: Observation | None, now: float,
                finished_at: datetime):
        self.correction = None
        self.angle = observation.angle if observation is not None else None
        if self.phase in ("align", "countdown"):
            if not aligned:
                self.phase = "align"
                self.feedback = "옆모습이 가이드에 들어오도록 위치를 조정하세요."
                self.countdown_frame_at = None
            elif self.phase == "align" or now - self.countdown_frame_at > 0.45:
                self.phase = "countdown"
                self.deadline = now + COUNTDOWN_SECONDS
                self.countdown_frame_at = now
                self.feedback = "위치 확인 완료. 자세를 유지하세요."
            else:
                self.countdown_frame_at = now
                if now >= self.deadline:
                    self._start_set(now)
            return

        if self.phase == "rest":
            if now >= self.deadline:
                if self.sets_completed == self.routine.sets:
                    self.exercise_index += 1
                    self._reset_exercise()
                    self.phase = "align"
                    self.feedback = "다음 운동의 가이드에 맞춰 주세요."
                else:
                    self._start_set(now)
            return

        if self.phase != "exercise":
            return

        update = self.counter.update(observation, now)
        self.feedback = update.feedback
        if update.correction:
            self.correction = self.visible_correction = update.correction
            self.correction_until = now + CORRECTION_DISPLAY_SECONDS
        self.interruptions += int(update.interrupted)
        if update.result is None:
            return
        if not update.result:
            self.rejections[update.correction.reason] += 1
            return
        self.reps += 1
        if self.reps != self.routine.reps:
            return

        self._end_set(now, completed=True)
        finished_exercise = self.sets_completed == self.routine.sets
        if finished_exercise:
            self._record(finished_at)
        if finished_exercise and self.exercise_index == len(self.routine.exercises) - 1:
            self.phase = "complete"
            self.feedback = "설정한 루틴을 모두 완료했어요."
        else:
            self.phase = "rest"
            self.deadline = now + self.routine.rest
            self.feedback = "다음 운동을 준비하세요." if finished_exercise else "다음 세트 전 휴식 중이에요."

    def snapshot(self, now: float) -> dict:
        wait = max(0, math.ceil(self.deadline - now)) if self.phase in ("rest", "countdown") else 0
        active = self.seconds + (now - self.set_started if self.set_started is not None else 0)
        set_number = self.sets_completed if self.phase in ("rest", "complete") else self.sets_completed + 1
        hint = self.visible_correction if self.phase == "exercise" and now < self.correction_until else None
        return {
            "phase": self.phase, "exercise": self.exercise.value, "index": self.exercise_index + 1,
            "items": len(self.routine.exercises), "set": min(set_number, self.routine.sets),
            "sets": self.routine.sets, "reps": self.reps, "target": self.routine.reps,
            "seconds": active, "angle": self.angle, "wait": wait,
            "feedback": hint.message if hint else self.feedback,
            "feedback_detail": hint.detail if hint else "",
            "correction": bool(hint),
            "last_set_report": self.last_set_report, "records": tuple(self.records),
        }


# MediaPipe 자세 인식과 카메라 가이드

WIDTH, HEIGHT = 960, 540
NOSE = 0
LEFT = (11, 13, 15, 23, 25, 27, 31)
RIGHT = tuple(index + 1 for index in LEFT)
CONNECTIONS = tuple(mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS)

GUIDES = {
    Exercise.SQUAT: ((.50, .16), (.50, .29), (.50, .49), (.50, .67), (.50, .85)),
    Exercise.PUSH_UP: ((.21, .46), (.31, .47), (.50, .58), (.65, .66), (.79, .73)),
    Exercise.SIT_UP: ((.23, .71), (.33, .72), (.51, .75), (.64, .54), (.76, .76)),
}
GUIDE_TOLERANCE = {
    Exercise.SQUAT: (.16, .17),
    Exercise.PUSH_UP: (.16, .19),
    Exercise.SIT_UP: (.16, .15),
}
GUIDE_LIMBS = {
    Exercise.SQUAT: (((.55, .41), (.51, .51)), ((.54, .89), (.56, .89))),
    Exercise.PUSH_UP: (((.32, .65), (.33, .88)), ((.79, .85), (.76, .89))),
    Exercise.SIT_UP: (((.42, .66), (.51, .68)), ((.79, .80), (.83, .80))),
}


@dataclass(frozen=True)
class Pose:
    image: np.ndarray
    world: np.ndarray
    confidence: np.ndarray

    def visible(self, indices):
        coords = self.image[list(indices), :2]
        return bool(
            np.all(self.confidence[list(indices)] >= VISIBILITY)
            and np.all(coords > .012) and np.all(coords < .988)
        )

    def side(self, exercise: Exercise):
        if exercise == Exercise.PUSH_UP:
            required = (0, 1, 2, 3, 4, 5)
        else:
            required = (0, 3, 4, 5)
        candidates = [side for side in (LEFT, RIGHT) if self.visible([side[i] for i in required])]
        return max(candidates, key=lambda side: np.mean(self.confidence[list(side)]), default=None)


class PoseDetector:
    def __init__(self, model_path: Path):
        # 모델을 메모리에서 읽어 Windows의 한글 경로에서도 동일하게 연다.
        self.landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(
            mp.tasks.vision.PoseLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_buffer=model_path.read_bytes()),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_poses=1,
            ),
        )
        self.previous_timestamp = -1

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.landmarker.close()

    def detect(self, frame: np.ndarray, now: float) -> Pose | None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp = max(self.previous_timestamp + 1, int(now * 1000))
        self.previous_timestamp = timestamp
        response = self.landmarker.detect_for_video(image, timestamp)
        if not response.pose_landmarks or not response.pose_world_landmarks:
            return None
        image_points = response.pose_landmarks[0]
        world_points = response.pose_world_landmarks[0]
        return Pose(
            np.array([(p.x, p.y, p.z) for p in image_points]),
            np.array([(p.x, p.y, p.z) for p in world_points]),
            np.array([min(p.visibility or 0.0, p.presence or 0.0) for p in image_points]),
        )


def fit_frame(raw: np.ndarray) -> np.ndarray:
    height, width = raw.shape[:2]
    scale = min(WIDTH / width, HEIGHT / height)
    fitted = cv2.resize(raw, (round(width * scale), round(height * scale)))
    canvas = np.full((HEIGHT, WIDTH, 3), (24, 22, 19), dtype=np.uint8)
    y, x = (HEIGHT - fitted.shape[0]) // 2, (WIDTH - fitted.shape[1]) // 2
    canvas[y:y+fitted.shape[0], x:x+fitted.shape[1]] = fitted
    return canvas


def observation(pose: Pose, exercise: Exercise, side: tuple[int, ...]) -> Observation | None:
    shoulder, elbow, wrist, hip, knee, ankle, toe = side
    world = pose.world
    pixels = pose.image[:, :2] * (WIDTH, HEIGHT)
    rule = RULES[exercise]
    fault = None
    if exercise == Exercise.PUSH_UP:
        angle = joint_angle(world[shoulder], world[elbow], world[wrist])
        body = pixels[ankle] - pixels[shoulder]
        length = max(np.linalg.norm(body), 1.0)
        waist = pixels[hip] - pixels[shoulder]
        below_line = waist[1] - (np.dot(waist, body) / length**2) * body[1]
        if below_line / length > .073:
            fault = PUSH_UP_SAG
    elif exercise == Exercise.SQUAT:
        angle = joint_angle(world[hip], world[knee], world[ankle])
        if angle < rule.departure and pose.visible((toe,)):
            foot = pixels[toe, 0] - pixels[ankle, 0]
            shin = max(np.linalg.norm(pixels[knee] - pixels[ankle]), 1.0)
            knee_past_toe = np.sign(foot) * (pixels[knee, 0] - pixels[toe, 0])
            if abs(foot) > shin * .20 and knee_past_toe > shin * .48:
                fault = SQUAT_KNEE
    else:
        torso = pixels[shoulder] - pixels[hip]
        if np.linalg.norm(torso) < 6:
            return None
        angle = math.degrees(math.atan2(-torso[1], abs(torso[0])))
    return Observation(angle, fault) if math.isfinite(angle) else None


def guide_points(exercise: Exercise, mirrored: bool):
    points = np.array(GUIDES[exercise])
    if mirrored:
        points[:, 0] = 1 - points[:, 0]
    return points


def evaluate_pose(pose: Pose | None, exercise: Exercise):
    if pose is None or (side := pose.side(exercise)) is None:
        return None, False, False
    shoulder, _, _, hip, knee, ankle, _ = side
    measured = observation(pose, exercise, side)
    indices = (NOSE, shoulder, hip, knee, ankle)
    mirrored = exercise != Exercise.SQUAT and pose.image[shoulder, 0] > pose.image[ankle, 0]
    if measured is None or not pose.visible(indices):
        return measured, False, mirrored

    points = pose.image[list(indices), :2]
    distances = (points - guide_points(exercise, mirrored)) / GUIDE_TOLERANCE[exercise]
    shoulder_width = np.linalg.norm((pose.image[11, :2] - pose.image[12, :2]) * (WIDTH, HEIGHT))
    torso = np.linalg.norm((pose.image[shoulder, :2] - pose.image[hip, :2]) * (WIDTH, HEIGHT))
    profile = torso > 20 and shoulder_width / torso <= .63
    rule = RULES[exercise]
    start_angle = measured.angle <= rule.start if rule.count_at_target else measured.angle >= rule.start
    aligned = start_angle and profile and bool(np.all(np.linalg.norm(distances, axis=1) <= 1))
    return measured, aligned, mirrored


def draw_guide(frame: np.ndarray, exercise: Exercise, aligned: bool, mirrored: bool = False):
    color = (146, 238, 117) if aligned else (216, 184, 139)
    points = np.round(guide_points(exercise, mirrored) * (WIDTH, HEIGHT)).astype(np.int32)
    limbs = np.array(GUIDE_LIMBS[exercise])
    if mirrored:
        limbs[:, :, 0] = 1 - limbs[:, :, 0]
    arms, feet = np.round(limbs * (WIDTH, HEIGHT)).astype(np.int32)
    lines = [points[1:], np.vstack((points[1], arms)), np.vstack((points[-1], feet))]
    faded = frame.copy()
    cv2.circle(faded, points[0], 28, color, -1, cv2.LINE_AA)
    cv2.polylines(faded, lines, False, color, 32, cv2.LINE_AA)
    cv2.addWeighted(faded, .16, frame, .84, 0, dst=frame)
    cv2.polylines(frame, lines, False, color, 2, cv2.LINE_AA)
    cv2.line(frame, points[0], points[1], color, 2, cv2.LINE_AA)
    for index, point in enumerate(points):
        cv2.circle(frame, point, 24 if index == 0 else 9, color, 2, cv2.LINE_AA)
    for x, direction in ((35, 1), (WIDTH - 35, -1)):
        for y, vertical in ((30, 1), (HEIGHT - 30, -1)):
            cv2.line(frame, (x, y), (x + 28 * direction, y), color, 2, cv2.LINE_AA)
            cv2.line(frame, (x, y), (x, y + 28 * vertical), color, 2, cv2.LINE_AA)


def draw_skeleton(frame: np.ndarray, pose: Pose):
    points = np.round(pose.image[:, :2] * (WIDTH, HEIGHT)).astype(np.int32)
    visible = [pose.visible([i]) for i in range(len(points))]
    for link in CONNECTIONS:
        if visible[link.start] and visible[link.end]:
            cv2.line(frame, points[link.start], points[link.end], (143, 235, 131), 2, cv2.LINE_AA)
    for index, active in enumerate(visible):
        if active:
            cv2.circle(frame, points[index], 4, (235, 255, 230), -1, cv2.LINE_AA)


def standby_frame() -> np.ndarray:
    frame = np.full((HEIGHT, WIDTH, 3), (35, 30, 23), dtype=np.uint8)
    for x in range(0, WIDTH, 32):
        cv2.line(frame, (x, 0), (x, HEIGHT), (46, 42, 35), 1)
    for y in range(0, HEIGHT, 32):
        cv2.line(frame, (0, y), (WIDTH, y), (46, 42, 35), 1)
    draw_guide(frame, Exercise.SQUAT, False)
    return frame


# CSV 저장과 통계

COLUMNS = [field.name for field in fields(Record)]
LEGACY_COLUMNS = COLUMNS[:-3]
RECORD_HEADERS = ["날짜", "운동", "성공 / 목표", "세트", "운동 시간", "자세 정확도", "이전 대비", "판정 근거"]
PALETTE = ("#536df0", "#13856b", "#aa5d1d")


def format_time(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def percentage(value):
    return f"{value:.1f}%" if pd.notna(value) else "—"


def accuracy_basis(successes: int, attempts: int) -> str:
    if not attempts:
        return "판정된 시도가 없어 정확도를 계산할 수 없어요."
    return f"성공 {successes}회 ÷ 판정 {attempts}회 = {percentage(successes / attempts * 100)}"


def read_reasons(value) -> dict[str, int]:
    if not isinstance(value, str) or not value:
        return {}
    try:
        reasons = json.loads(value)
    except ValueError:
        return {}
    if not isinstance(reasons, dict):
        return {}
    return {reason: count for reason, count in reasons.items() if type(count) is int and count > 0}


def reason_counts(reasons) -> str:
    return " · ".join(f"{reason} {count}회" for reason, count in reasons)


def excluded_counts(interruptions: int, unfinished: int) -> str:
    parts = []
    if interruptions:
        parts.append(f"추적이 끊긴 시도 {interruptions}회")
    if unfinished:
        parts.append(f"종료 시 진행 중이던 시도 {unfinished}회")
    return " · ".join(parts)


class History:
    def __init__(self, path: Path):
        self.path = path
        self.lock = Lock()

    def save(self, records: list[Record]):
        if not records:
            return
        with self.lock:
            rows = []
            if self.path.exists():
                with self.path.open(encoding="utf-8-sig", newline="") as stream:
                    reader = csv.DictReader(stream)
                    if reader.fieldnames not in (COLUMNS, LEGACY_COLUMNS):
                        raise ValueError("기록 파일의 열 구성이 올바르지 않습니다.")
                    rows = list(reader)
            seen = {(row["session_id"], row["exercise"]) for row in rows}
            rows.extend(asdict(r) for r in records if (r.session_id, r.exercise) not in seen)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8-sig", newline="", delete=False,
                    dir=self.path.parent, suffix=".tmp",
                ) as stream:
                    temp_path = Path(stream.name)
                    writer = csv.DictWriter(stream, fieldnames=COLUMNS)
                    writer.writeheader()
                    writer.writerows(rows)
                temp_path.replace(self.path)
            finally:
                if temp_path is not None:
                    temp_path.unlink(missing_ok=True)

    def read(self) -> pd.DataFrame:
        with self.lock:
            if not self.path.exists():
                frame = pd.DataFrame(columns=COLUMNS)
            else:
                frame = pd.read_csv(self.path, encoding="utf-8-sig")
        for column in COLUMNS[len(LEGACY_COLUMNS):]:
            if column not in frame:
                frame[column] = None
        frame["date"] = frame["finished_at"].astype(str).str.slice(0, 10)
        frame["accuracy"] = frame["successful_reps"] * 100 / frame["judged_attempts"].replace(0, np.nan)
        frame["complete"] = frame["sets_completed"] == frame["sets_target"]
        frame["pace"] = (frame["active_seconds"] / frame["sets_target"]).where(frame["complete"])
        baseline = frame.groupby(["exercise", "target_reps", "sets_target"])["pace"].transform(
            lambda values: values.ffill().shift(),
        )
        frame["improvement"] = (baseline - frame["pace"]) * 100 / baseline.replace(0, np.nan)
        return frame


def today_records(frame: pd.DataFrame):
    return frame[frame["date"] == datetime.now().date().isoformat()]


def summary(frame: pd.DataFrame):
    today = today_records(frame)
    attempts = today["judged_attempts"].sum()
    successes = today["successful_reps"].sum()
    accuracy = percentage(successes / attempts * 100) if attempts else "—"
    changes = frame["improvement"].dropna()
    improvement = f"{changes.iloc[-1]:+.1f}%" if len(changes) else "—"
    return format_time(today["active_seconds"].sum()), f"{successes}회", accuracy, improvement


def _record_basis(row) -> str:
    basis = accuracy_basis(row.successful_reps, row.judged_attempts)
    reasons = read_reasons(row.rejection_reasons)
    if reasons:
        basis += f" · {reason_counts(reasons.items())}"
    elif row.judged_attempts > row.successful_reps:
        basis += f" · 불인정 {row.judged_attempts - row.successful_reps}회 (사유 기록 없음)"
    excluded = excluded_counts(
        int(row.tracking_interruptions) if pd.notna(row.tracking_interruptions) else 0,
        int(row.unfinished_attempts) if pd.notna(row.unfinished_attempts) else 0,
    )
    return f"{basis} · {excluded}는 판정에서 제외" if excluded else basis


def history_rows(frame: pd.DataFrame):
    return [
        [
            row.date, row.exercise, f"{row.successful_reps} / {row.target_reps}",
            f"{row.sets_completed} / {row.sets_target}", format_time(row.active_seconds),
            percentage(row.accuracy), f"{row.improvement:+.1f}%" if pd.notna(row.improvement) else "—",
            _record_basis(row),
        ]
        for row in frame.tail(30).iloc[::-1].itertuples(index=False)
    ]


def _axes():
    figure = Figure(figsize=(4.5, 2.25), dpi=120, layout="constrained", facecolor="#ffffff")
    ax = figure.subplots()
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#eaedf2", linewidth=.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#e1e5ec")
    ax.tick_params(axis="both", length=0, labelsize=8, colors="#5e6880", pad=7)
    ax.margins(x=.12)
    return figure, ax


def _font():
    for family in ("Malgun Gothic", "AppleGothic", "Noto Sans CJK KR"):
        try:
            font_manager.findfont(family, fallback_to_default=False)
        except ValueError:
            continue
        return font_manager.FontProperties(family=family, size=8)
    return font_manager.FontProperties(size=8)


def _line_chart(series: pd.Series, color, label: str):
    figure, ax = _axes()
    if series.empty:
        ax.set_ylim(0, 10)
        ax.set_xticks([])
        return figure
    recent = series.sort_index().tail(7)
    labels = [date[5:] for date in recent.index]
    x = np.arange(len(labels))
    ax.plot(x, recent, color=color, linewidth=2.2, marker="o", markersize=5)
    ax.fill_between(x, recent.astype(float), color=color, alpha=.09)
    ax.set_xticks(x, labels)
    ax.set_ylim(bottom=0)
    ax.set_ylabel(label, fontproperties=_font(), color="#5e6880")
    return figure


def charts(frame: pd.DataFrame):
    daily_minutes = frame.groupby("date")["active_seconds"].sum() / 60
    completed = frame[frame["complete"]]
    pace_label = "완료 기록이 생기면 변화가 표시됩니다."
    pace = pd.Series(dtype=float)
    if not completed.empty:
        last = completed.iloc[-1]
        comparable = completed[
            (completed["exercise"] == last["exercise"])
            & (completed["target_reps"] == last["target_reps"])
            & (completed["sets_target"] == last["sets_target"])
        ]
        pace = comparable.groupby("date")["pace"].mean()
        pace_label = f"{last['exercise']} · 세트당 {last['target_reps'] // last['sets_target']}회 × {last['sets_target']}세트"

    accuracy_figure, ax = _axes()
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    counts = frame.groupby("exercise")[["successful_reps", "judged_attempts"]].sum()
    order = [exercise.value for exercise in Exercise]
    rates = (counts["successful_reps"] * 100 / counts["judged_attempts"].replace(0, np.nan)).reindex(order)
    korean_font = _font()
    labels = order if korean_font.get_name() != "DejaVu Sans" else ["Squat", "Push-up", "Sit-up"]
    ax.set_xlim(-.5, len(order) - .5)
    ax.set_xticks(range(len(order)), labels, fontproperties=korean_font)
    for index, rate in enumerate(rates):
        if pd.notna(rate):
            ax.bar(index, rate, width=.44, color=PALETTE[index])
            ax.annotate(f"{rate:.0f}%", (index, rate), xytext=(0, -12 if rate >= 30 else 5),
                        textcoords="offset points", ha="center", fontsize=8,
                        color="white" if rate >= 30 else PALETTE[index])
    return (
        _line_chart(daily_minutes, PALETTE[0], "min"),
        _line_chart(pace, PALETTE[1], "sec"),
        accuracy_figure,
        pace_label,
    )


# 웹캠 세션과 음성 안내

HEARTBEAT_TIMEOUT = 35
SPEECH_INTERVAL = 4.0


def _encode(frame):
    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 79])
    return buffer.tobytes() if ok else b""


class VoiceCoach:
    def __init__(self):
        self.messages = Queue(maxsize=1)
        self.enabled = Event()
        self.status = "음성 준비 중"
        self.thread = Thread(target=self._speak, daemon=True, name="voice-coach")
        self.thread.start()

    def say(self, message: str):
        if self.enabled.is_set() and self.status == "한국어 음성 켜짐":
            try:
                self.messages.put_nowait(message)
            except Full:
                pass

    def _speak(self):
        try:
            import pyttsx3

            engine = pyttsx3.init()
            korean = next((
                voice for voice in engine.getProperty("voices")
                if any(
                    token in str((voice.id, voice.name, voice.languages)).lower()
                    for token in ("ko-kr", "ko_kr", "korean", "한국어", "'412'", "'0412'", "heami")
                )
            ), None)
            if korean is None:
                self.status = "이 PC에 한국어 음성이 설치되어 있지 않아요."
                return
            engine.setProperty("voice", korean.id)
            self.status = "한국어 음성 켜짐"
            while True:
                try:
                    message = self.messages.get(timeout=1)
                except Empty:
                    continue
                if self.enabled.is_set():
                    engine.say(message)
                    engine.runAndWait()
        except Exception:
            LOG.exception("음성 안내를 사용할 수 없습니다.")
            self.status = "이 PC에서 음성을 재생할 수 없어요."


class CameraTrainer:
    def __init__(self, model: Path, history: History, source: int = 0):
        self.model, self.history, self.source = model, history, source
        self.lock = RLock()
        self.stop_event = Event()
        self.thread = None
        self.workout = None
        self.voice = None
        self.opening = False
        self.error = ""
        self.saved_count = 0
        self.save_error = ""
        self.revision = 0
        self.last_poll = 0.0
        self.standby = _encode(standby_frame())
        self.frame = (0, self.standby)

    def start(self, routine: Routine, voice: bool):
        with self.lock:
            if self.thread is not None and self.thread.is_alive():
                raise ValueError("카메라가 사용 중이에요. 잠시 후 다시 시작하세요.")
            if self.save_error:
                raise ValueError("먼저 기록 저장을 다시 시도하세요.")
            self.workout = Workout(routine)
            self.saved_count = 0
            self.last_poll = time.monotonic()
            self.stop_event.clear()
            self.error = ""
            self.opening = True
            if voice and self.voice is None:
                self.voice = VoiceCoach()
            if self.voice is not None:
                self.voice.enabled.set() if voice else self.voice.enabled.clear()
            self.thread = Thread(target=self._run, daemon=True, name="local-camera")
            self.thread.start()

    def stop(self):
        with self.lock:
            self.stop_event.set()
            if self.voice is not None:
                self.voice.enabled.clear()

    def shutdown(self):
        self.stop()
        if self.thread is not None:
            self.thread.join(timeout=3)

    def _publish(self, jpeg: bytes):
        if jpeg:
            with self.lock:
                self.frame = self.frame[0] + 1, jpeg

    def read_frame(self):
        with self.lock:
            return self.frame

    def snapshot(self):
        with self.lock:
            now = time.monotonic()
            self.last_poll = now
            data = self.workout.snapshot(now) if self.workout else {"phase": "idle"}
            if data.get("exercise"):
                data["joint"] = RULES[self.workout.exercise].joint
            busy = self.thread is not None and self.thread.is_alive()
            if self.error:
                data.update(phase="error", feedback=self.error, feedback_detail="", correction=False)
            elif busy and self.stop_event.is_set():
                data.update(phase="stopping", feedback="카메라를 끄고 기록을 저장하고 있어요.",
                            feedback_detail="", correction=False)
            elif self.opening:
                data.update(phase="opening", feedback="카메라와 자세 인식 모델을 준비하고 있어요.")
            voice_status = ""
            if self.voice is not None and self.voice.enabled.is_set():
                voice_status = self.voice.status
            return dict(data, busy=busy, revision=self.revision,
                        save_error=self.save_error, voice_status=voice_status)

    def retry_save(self):
        with self.lock:
            self._save()

    def _save(self):
        records = self.workout.records[self.saved_count:] if self.workout else []
        if records:
            try:
                self.history.save(records)
                self.saved_count += len(records)
                self.revision += 1
                self.save_error = ""
            except (OSError, ValueError):
                LOG.exception("CSV 저장 실패")
                self.save_error = "기록을 저장하지 못했어요. CSV를 열어 둔 프로그램을 닫고 다시 시도하세요."

    def _still_running(self):
        with self.lock:
            return not self.stop_event.is_set() and time.monotonic() - self.last_poll < HEARTBEAT_TIMEOUT

    def _run(self):
        capture = None
        try:
            if not self.model.is_file():
                raise RuntimeError("자세 인식 모델 파일이 없어요. models 폴더의 모델 파일을 확인하세요.")
            capture = cv2.VideoCapture(self.source)
            if not capture.isOpened():
                raise RuntimeError("카메라를 열 수 없어요. 연결, Windows 카메라 권한, 다른 앱의 카메라 사용을 확인하세요.")
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
            with PoseDetector(self.model) as detector:
                with self.lock:
                    self.opening = False
                self._coach(capture, detector)
        except RuntimeError as error:
            LOG.exception("카메라 또는 모델 초기화 실패")
            with self.lock:
                self.error = str(error) if any("가" <= char <= "힣" for char in str(error)) else (
                    "카메라 또는 자세 인식 모델을 실행할 수 없어요. 터미널의 오류를 확인하세요."
                )
        except Exception:
            LOG.exception("카메라 루프 실패")
            with self.lock:
                self.error = "영상 처리 중 오류가 발생했어요. 터미널의 오류를 확인하세요."
        finally:
            if capture is not None:
                capture.release()
            self._publish(self.standby)
            with self.lock:
                self.opening = False
                if self.voice is not None:
                    self.voice.enabled.clear()
                self.workout.stop(time.monotonic(), datetime.now().astimezone())
                self._save()

    def _coach(self, capture, detector):
        last_spoken = -float("inf")
        failed_reads = 0
        while self._still_running():
            ok, raw = capture.read()
            if not ok:
                failed_reads += 1
                if failed_reads >= 20:
                    raise RuntimeError("카메라 연결이 끊겼어요. 연결을 확인하고 다시 시작하세요.")
                self.stop_event.wait(.05)
                continue
            failed_reads = 0
            frame = fit_frame(raw)
            now = time.monotonic()
            pose = detector.detect(frame, now)
            exercise = self.workout.exercise
            measured, aligned, mirrored = evaluate_pose(pose, exercise)
            with self.lock:
                if self.stop_event.is_set():
                    break
                self.workout.advance(aligned, measured, now, datetime.now().astimezone())
                phase, correction = self.workout.phase, self.workout.correction
                if not self.save_error:
                    self._save()
            if phase == "complete":
                break
            if correction and self.voice is not None and now - last_spoken >= SPEECH_INTERVAL:
                self.voice.say(correction.message)
                last_spoken = now
            if pose is not None:
                draw_skeleton(frame, pose)
            if phase in ("align", "countdown"):
                changed = exercise != self.workout.exercise
                draw_guide(frame, self.workout.exercise, aligned and not changed, mirrored and not changed)
            self._publish(_encode(frame))


# Gradio 대시보드

STYLE = """
:root { color-scheme: light; }
body, .gradio-container {
    background: #f1f3f7 !important;
}
.gradio-container {
    --body-text-color: #172039;
    --block-label-text-color: #35405b;
    --body-text-color-subdued: #59647d;
    --block-background-fill: #fff;
    --panel-background-fill: #fff;
    --background-fill-primary: #fff;
    --background-fill-secondary: #edf1f7;
    --input-background-fill: #f7f8fb;
    max-width: 1328px !important;
    padding: 26px 28px 30px !important;
    margin: auto;
}
footer { display: none !important; }
#intro {
    background: #171f37;
    border: 1px solid #222e4e;
    border-radius: 22px;
    padding: 23px 28px 25px;
}
#intro h1 { font-size: clamp(24px, 4vw, 38px); font-weight: 800; letter-spacing: -.055em; color: #fff; margin: 0; }
#intro p { color: #c0cae3; margin: 7px 0 0; font-size: 14px; }
#intro h1 strong { color: #c1f87c; }
#routine, #coach, #insights, #camera-card {
    background: #fff;
    border: 1px solid #e0e4ee;
    border-radius: 18px;
    padding: 18px 20px;
    box-shadow: 0 3px 9px #21325705;
}
#routine { padding-bottom: 15px; }
#routine-heading h2, #insights-heading h2 {
    font-size: 17px; letter-spacing: -.035em; margin: 0;
}
#routine-settings { align-items: start; }
#camera-card, #coach { min-width: 0 !important; }
#workspace.pending, #workspace .pending, #insights .pending { opacity: 1 !important; }
#camera-image { background: #171e23; overflow: hidden; border-radius: 11px; }
#camera-image img { display: block; width: 100%; aspect-ratio: 16 / 9; object-fit: contain; }
#camera-note p, #detail-note p, #voice-status p, #plot-caption p { color: #59647d; font-size: 12px; margin: 0; line-height: 1.5; }
#phase { width: max-content; padding: 5px 10px; background: #eef0fd; border-radius: 24px; }
#phase p { margin: 0; color: #3d4b9e; font-size: 16px; font-weight: 800; }
#live-count p, .stat p { margin: 0; color: #53617b; }
#live-count h2 { font-size: clamp(38px, 5vw, 64px); margin: 0; color: #192543; line-height: 1.15; letter-spacing: -.065em; }
#live-count h2 strong { color: #4861db; }
#track { min-height: 21px; }
#track p { margin: 0; font-size: 12px; font-weight: 700; }
#feedback .coaching-message {
    display: flex; flex-direction: column; justify-content: center; gap: 4px;
    min-height: 110px; padding: 14px clamp(14px, 2.2vw, 27px);
    border: 2px solid #1b2541; border-radius: 12px; background: #1b2541; color: #fff;
}
#feedback .coaching-message[data-tone="correction"] {
    border-color: #f7c76a; background: #3a290d; color: #fff2d4;
}
#feedback .coaching-message :is(.coaching-label, .coaching-cue, .coaching-detail) {
    color: inherit !important;
}
#feedback .coaching-label { font-size: 14px; font-weight: 700; }
#feedback .coaching-cue {
    font-size: clamp(26px, 3.15vw, 48px); font-weight: 800; letter-spacing: -.035em; line-height: 1.15;
    word-break: keep-all; overflow-wrap: anywhere;
}
#feedback .coaching-detail { font-size: clamp(16px, 1.45vw, 22px); line-height: 1.35; }
#feedback-review {
    padding: 16px 18px; border: 1px solid #ccddd8; border-radius: 12px; background: #f0f7f4;
}
#feedback-review h3 { font-size: clamp(22px, 2.1vw, 30px); margin: 0 0 7px; }
#feedback-review h4 { font-size: 19px; margin: 12px 0 2px; }
#feedback-review p, #feedback-review li { font-size: 16px; line-height: 1.55; }
#today-basis p { font-size: 14px; color: #42516d; margin: 0; }
#live-meter-row { border-top: 1px solid #edf0f5; padding-top: 10px; }
.stat {
    flex: 1; min-width: 126px !important; padding: 9px 12px;
    background: #f4f6fa; border: 1px solid #edf0f6; border-radius: 10px;
}
.stat p { font-size: 12px; }
.stat h3 { margin: 0; font-size: 23px; font-weight: 800; letter-spacing: -.04em; }
#start { background: #4359d4; border-color: #4359d4; color: #fff; }
#start:hover { background: #324ac4; }
#start:disabled { opacity: .58; }
.chart { min-width: 200px !important; }
.chart .prose h3 { font-size: 14px; margin: 0; }
.chart .block { border-color: #edf0f5 !important; box-shadow: none; }
@media (max-width: 720px) {
    .gradio-container { padding: 12px !important; }
    #intro { padding: 18px; }
    #routine, #coach, #insights, #camera-card { padding: 14px; }
    #workspace { flex-direction: column; }
    #history-grid th span { white-space: normal !important; overflow-wrap: anywhere; }
}
"""

PHASES = {
    "idle": "시작 전", "opening": "카메라 준비 중", "align": "위치 맞추기",
    "countdown": "시작 카운트다운", "exercise": "운동 진행 중", "rest": "휴식 중",
    "complete": "루틴 완료", "stopping": "저장 중", "stopped": "운동 종료",
    "error": "카메라 확인",
}

CAMERA_IMAGE = '<img src="{}" width="960" height="540" alt="웹캠과 전신 위치 가이드">'
STANDBY_VIEW = CAMERA_IMAGE.format("/standby.jpg")
LIVE_VIEW = CAMERA_IMAGE.format("/camera.mjpg")


def _stat(label: str, value: str):
    return f"{label}\n\n### {value}"


def _integer(value, label):
    if value is None or not math.isfinite(value) or int(value) != value:
        raise ValueError(f"{label}은 정수로 입력하세요.")
    return int(value)


def _coaching_banner(message, detail="", corrective=False):
    tone = "correction" if corrective else "normal"
    heading = "자세 교정" if corrective else "실시간 코칭"
    detail_html = f'<span class="coaching-detail">{escape(detail)}</span>' if detail else ""
    return (
        f'<div class="coaching-message" data-tone="{tone}" role="status" aria-live="polite">'
        f'<span class="coaching-label">{heading}</span>'
        f'<strong class="coaching-cue">{escape(message)}</strong>{detail_html}</div>'
    )


def _review_detail(exercise, successes, attempts, reasons, interruptions, unfinished):
    lines = [f"**{accuracy_basis(successes, attempts)}**", exercise_criteria(exercise)]
    reasons = sorted(reasons, key=lambda item: item[1], reverse=True)
    if reasons:
        lines.append("불인정 사유")
        lines.append("\n".join(f"- {reason} **{count}회**" for reason, count in reasons))
        advice = correction_advice(exercise, reasons[0][0])
        if advice:
            lines.append(f"**다음 동작** · {advice}")
    elif attempts:
        lines.append("판정된 시도는 모두 기준을 충족했어요.")
    excluded = excluded_counts(interruptions, unfinished)
    if excluded:
        lines.append(f"{excluded}는 정확도 계산에서 제외했어요.")
    return "\n\n".join(lines)


def _workout_review(data):
    phase = data["phase"]
    if phase == "rest" and (report := data.get("last_set_report")):
        detail = _review_detail(
            report.exercise, report.successful_reps, report.judged_attempts, report.reasons,
            report.tracking_interruptions, report.unfinished_attempts,
        )
        return f"### {report.exercise.value} {report.number}세트 요약\n\n{detail}", False
    records = data.get("records", ())
    if phase not in ("complete", "stopped", "error") or not records:
        return "", False
    parts = ["### 이번 운동 결과"]
    if len(records) > 1:
        successes = sum(record.successful_reps for record in records)
        attempts = sum(record.judged_attempts for record in records)
        parts.append(f"**전체 정확도 · {accuracy_basis(successes, attempts)}**")
    for record in records:
        details = _review_detail(
            Exercise(record.exercise), record.successful_reps, record.judged_attempts,
            read_reasons(record.rejection_reasons).items(), record.tracking_interruptions,
            record.unfinished_attempts,
        )
        parts.append(f"#### {record.exercise}\n\n{details}")
    return "\n\n".join(parts), True


def _live(data):
    phase = data["phase"]
    active = data["busy"]
    exercise = data.get("exercise")
    count = f"## **{data['reps']}** / {data['target']}" if exercise else "## **0** / —"
    track = f"{exercise} · 종목 {data['index']}/{data['items']} · 세트 {data['set']}/{data['sets']}" if exercise else "루틴을 설정하고 시작하세요."
    label = PHASES[phase]
    if phase in ("countdown", "rest"):
        label += f" · {data['wait']}초"
    angle = data.get("angle")
    joint = f"{data['joint']} 각도" if exercise else "관절 각도"
    angle_text = "—" if angle is None or phase != "exercise" else f"{angle:.0f}°"
    coach = data.get("feedback") or "운동을 시작하면 가이드가 표시됩니다."
    review, show_final_review = _workout_review(data)
    save_error = data["save_error"]
    settings_update = gr.update(interactive=not active)
    return (
        f"**{label}**", count, track,
        _coaching_banner(coach, data.get("feedback_detail", ""), data.get("correction", False)),
        _stat("운동 시간", format_time(data.get("seconds", 0))),
        _stat(joint, angle_text),
        data["voice_status"], save_error,
        gr.update(visible=bool(save_error) and not active),
        gr.update(interactive=not active and not save_error),
        gr.update(interactive=active and phase != "stopping"),
        settings_update, settings_update, settings_update, settings_update, settings_update,
        gr.update(active=active),
        gr.update(value=LIVE_VIEW if phase in ("align", "countdown", "exercise", "rest") else STANDBY_VIEW,
                  visible=not show_final_review),
        gr.update(value=review, visible=bool(review)),
        gr.update(visible=not show_final_review),
    )


def build_dashboard(trainer, history):
    theme = gr.themes.Soft(
        primary_hue="indigo", neutral_hue="slate",
        font=["Inter", "Malgun Gothic", "system-ui", "sans-serif"],
    ).set(
        body_background_fill="#f1f3f7", body_background_fill_dark="#f1f3f7",
        body_text_color="#172039", body_text_color_dark="#172039",
        background_fill_primary="#ffffff", background_fill_primary_dark="#ffffff",
        background_fill_secondary="#edf1f7", background_fill_secondary_dark="#edf1f7",
        block_background_fill="#ffffff", block_background_fill_dark="#ffffff",
        block_border_color="#e0e4ee", block_border_color_dark="#e0e4ee",
        input_background_fill="#f7f8fb", input_background_fill_dark="#f7f8fb",
        input_border_color="#dce2ee", input_border_color_dark="#dce2ee",
    )

    def refresh(last_revision, force=False):
        data = trainer.snapshot()
        live = _live(data)
        version = data["revision"]
        if not force and last_revision == version:
            return (*live, *[gr.skip() for _ in range(11)], version)
        error = ""
        try:
            frame = history.read()
        except (OSError, ValueError, KeyError, TypeError):
            LOG.exception("CSV 읽기 실패")
            error = "운동 기록을 읽을 수 없어요. CSV 형식과 파일 사용 상태를 확인하세요."
            frame = pd.DataFrame(columns=[
                "date", "judged_attempts", "successful_reps", "active_seconds",
                "improvement", "complete", "exercise",
            ])
        duration, successes, accuracy, improvement = summary(frame)
        today = today_records(frame)
        today_basis = accuracy_basis(today["successful_reps"].sum(), today["judged_attempts"].sum())
        minutes, pace, scores, pace_note = charts(frame)
        return (
            *live, _stat("오늘 운동 시간", duration), _stat("오늘 성공 횟수", successes),
            _stat("오늘 자세 정확도", accuracy), _stat("최근 시간 단축률", improvement),
            f"**오늘 정확도 계산** · {today_basis}", minutes, pace, scores, pace_note, history_rows(frame),
            error or ("아직 저장된 운동 기록이 없어요." if frame.empty else ""), version,
        )

    def start_workout(selected, rep_goal, set_goal, rest_goal, audio, version):
        try:
            routine = Routine(
                tuple(Exercise(value) for value in (selected or [])),
                _integer(rep_goal, "횟수"), _integer(set_goal, "세트"),
                _integer(rest_goal, "휴식"),
            )
            trainer.start(routine, audio)
        except ValueError as error:
            gr.Warning(str(error))
        return refresh(version)

    def stop_workout(version):
        trainer.stop()
        return refresh(version)

    def retry(version):
        trainer.retry_save()
        return refresh(version, force=True)

    with gr.Blocks(
        title="AI PT 트레이너", theme=theme, css=STYLE, analytics_enabled=False,
        fill_width=True,
    ) as dashboard:
        revision = gr.State(-1)
        timer = gr.Timer(.35, active=False)
        gr.Markdown(
            "# **AI PT** 트레이너\n\n웹캠으로 자세를 확인하고 운동 기록을 쌓아보세요.",
            elem_id="intro",
        )
        with gr.Column(elem_id="routine"):
            gr.Markdown("## 루틴 설정", elem_id="routine-heading")
            with gr.Row(elem_id="routine-settings"):
                exercises = gr.Dropdown(
                    [exercise.value for exercise in Exercise], value=[Exercise.SQUAT.value],
                    multiselect=True, max_choices=3, label="운동 순서",
                    info="선택한 순서로 진행됩니다.", scale=4, min_width=220,
                )
                reps = gr.Number(10, minimum=1, maximum=99, label="세트당 횟수", min_width=110)
                sets = gr.Number(3, minimum=1, maximum=20, label="세트", min_width=98)
                rest = gr.Number(30, minimum=0, maximum=300, label="휴식 시간 (초)", min_width=120)
            with gr.Row():
                voice = gr.Checkbox(label="한국어 음성 안내", value=True, scale=3, min_width=140)
                stop_button = gr.Button("운동 종료 및 저장", interactive=False, min_width=150)
                start_button = gr.Button("운동 시작", variant="primary", min_width=120, elem_id="start")
        with gr.Row(elem_id="workspace", equal_height=False):
            with gr.Column(scale=7, elem_id="camera-card", min_width=280):
                feedback = gr.HTML(
                    _coaching_banner("운동을 시작하면 가이드가 표시됩니다."), elem_id="feedback",
                )
                review = gr.Markdown(visible=False, elem_id="feedback-review")
                camera_view = gr.HTML(STANDBY_VIEW, elem_id="camera-image")
                camera_note = gr.Markdown(
                    "카메라를 옆에 두고 머리부터 발까지 화면에 담아 주세요. 영상은 이 PC에서만 처리됩니다.",
                    elem_id="camera-note",
                )
            with gr.Column(scale=3, elem_id="coach", min_width=245):
                phase = gr.Markdown("**시작 전**", elem_id="phase")
                count = gr.Markdown("## **0** / —", elem_id="live-count")
                track = gr.Markdown("루틴을 설정하고 시작하세요.", elem_id="track")
                with gr.Row(elem_id="live-meter-row"):
                    elapsed = gr.Markdown(_stat("운동 시간", "00:00"), elem_classes=["stat"])
                    angle = gr.Markdown(_stat("관절 각도", "—"), elem_classes=["stat"])
                voice_status = gr.Markdown(elem_id="voice-status")
                save_message = gr.Markdown()
                retry_button = gr.Button("기록 다시 저장", visible=False)
        with gr.Column(elem_id="insights"):
            gr.Markdown("## 운동 통계", elem_id="insights-heading")
            with gr.Row():
                total = gr.Markdown(elem_classes=["stat"])
                successes = gr.Markdown(elem_classes=["stat"])
                accuracy = gr.Markdown(elem_classes=["stat"])
                improvement = gr.Markdown(elem_classes=["stat"])
            accuracy_explanation = gr.Markdown(elem_id="today-basis")
            with gr.Row():
                with gr.Column(elem_classes=["chart"]):
                    gr.Markdown("### 날짜별 운동 시간")
                    daily = gr.Plot(show_label=False)
                with gr.Column(elem_classes=["chart"]):
                    gr.Markdown("### 세트 완료 시간")
                    pace = gr.Plot(show_label=False)
                    pace_note = gr.Markdown(elem_id="plot-caption")
                with gr.Column(elem_classes=["chart"]):
                    gr.Markdown("### 종목별 자세 정확도")
                    scores = gr.Plot(show_label=False)
            gr.Markdown("### 최근 운동 기록")
            records = gr.Dataframe(
                headers=RECORD_HEADERS, datatype=["str"] * len(RECORD_HEADERS),
                interactive=False, show_label=False, wrap=True, max_height=330, elem_id="history-grid",
                column_widths=["11%", "12%", "10%", "7%", "9%", "10%", "9%", "32%"],
            )
            read_error = gr.Markdown()
            gr.Markdown(
                "성공과 불인정을 합쳐 판정 횟수를 계산합니다. 정확도는 관절 각도와 자세 기준의 충족률입니다.",
                elem_id="detail-note",
            )

        targets = [
            phase, count, track, feedback, elapsed, angle, voice_status, save_message, retry_button,
            start_button, stop_button, exercises, reps, sets, rest, voice, timer, camera_view,
            review, camera_note, total, successes, accuracy, improvement, accuracy_explanation,
            daily, pace, scores, pace_note, records,
            read_error, revision,
        ]
        dashboard.load(lambda: refresh(-1, force=True), outputs=targets, api_name=False)
        timer.tick(refresh, inputs=[revision], outputs=targets, trigger_mode="once",
                   show_progress="hidden", api_name=False)
        start_button.click(start_workout, [exercises, reps, sets, rest, voice, revision],
                           targets, api_name=False)
        stop_button.click(stop_workout, [revision], targets, api_name=False)
        retry_button.click(retry, [revision], targets, api_name=False)
    return dashboard


# 로컬 웹 서버

def create_app(camera=0, history_path=None, browser_address=None):
    history = History(history_path or APP_HOME / "data" / "workouts.csv")
    trainer = CameraTrainer(MODEL_PATH, history, camera)

    @asynccontextmanager
    async def lifespan(_):
        browser = None
        if browser_address:
            browser = Timer(.9, webbrowser.open, args=(browser_address,))
            browser.daemon = True
            browser.start()
        try:
            yield
        finally:
            if browser is not None:
                browser.cancel()
            trainer.shutdown()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])

    async def camera_frames():
        last_version = -1
        last_sent = 0.0
        while True:
            version, jpeg = trainer.read_frame()
            now = monotonic()
            if version != last_version or now - last_sent >= .7:
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                    + str(len(jpeg)).encode("ascii") + b"\r\n\r\n" + jpeg + b"\r\n"
                )
                last_version = version
                last_sent = now
            await asyncio.sleep(.055)

    @app.get("/standby.jpg", include_in_schema=False)
    async def standby():
        return Response(trainer.standby, media_type="image/jpeg")

    @app.get("/camera.mjpg", include_in_schema=False)
    async def stream(request: Request):
        if request.headers.get("sec-fetch-site") == "cross-site":
            raise HTTPException(status_code=403)
        return StreamingResponse(
            camera_frames(), media_type="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store", "Cross-Origin-Resource-Policy": "same-origin"},
        )

    return gr.mount_gradio_app(app, build_dashboard(trainer, history), path="/", show_error=False)


def main(options):
    # 환경을 다시 만들지 않고, 모델만 매 실행 시 확인합니다.
    try:
        prepare_model()
    except (OSError, ValueError) as error:
        print(f"자세 모델을 준비하지 못했습니다. {error}", file=sys.stderr, flush=True)
        return 1

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        try:
            listener.bind(("127.0.0.1", options.port))
        except OSError as error:
            print(f"로컬 서버를 열지 못했습니다. {error}", file=sys.stderr)
            return 1
        port = listener.getsockname()[1]
        address = f"http://127.0.0.1:{port}"
        app = create_app(camera=options.camera, browser_address=None if options.no_browser else address)
        config = uvicorn.Config(
            app, host="127.0.0.1", port=port, access_log=False, timeout_graceful_shutdown=2,
        )
        print(f"AI PT 트레이너 주소 {address}", flush=True)
        uvicorn.Server(config).run(sockets=[listener])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(parse_options()))
