from dataclasses import dataclass
from enum import StrEnum


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
    view_guide: str
    cue: str
    at_target: str
    shallow: str
    count_at_target: bool = False


# 종목별 판정 기준
# - 스쿼트·팔굽혀펴기: 웹캠 정면(FRONT)
# - 윗몸일으키기: 웹캠 오른쪽 측면(RIGHT-SIDE)
RULES: dict[Exercise, Rule] = {
    Exercise.SQUAT: Rule(
        start=155.0,
        departure=145.0,
        target=115.0,
        joint="무릎",
        view_guide="정면",
        cue="무릎을 굽혀 아래로 내려가세요.",
        at_target="천천히 일어서세요.",
        shallow="무릎을 조금 더 굽혀 보세요.",
        count_at_target=False,
    ),
    Exercise.PUSH_UP: Rule(
        start=140.0,
        departure=135.0,
        target=120.0,
        joint="팔꿈치",
        view_guide="정면",
        cue="팔꿈치를 굽혀 가슴을 낮추세요.",
        at_target="바닥을 밀어 시작 자세로 돌아오세요.",
        shallow="팔꿈치를 조금 더 굽혀 보세요.",
        count_at_target=False,
    ),
    Exercise.SIT_UP: Rule(
        start=108.0,
        departure=98.0,
        target=75.0,
        joint="고관절",
        view_guide="오른쪽 측면",
        cue="반동 없이 상체를 일으켜 보세요.",
        at_target="상체를 천천히 내려 누운 자세로 돌아오세요.",
        shallow="상체를 조금 더 일으켜 보세요.",
        count_at_target=True,
    ),
}

VISIBILITY = 0.45
MAX_FRAME_GAP = 0.38
POSITION_HOLD = 0.12
FAULT_HOLD = 0.20


@dataclass(frozen=True)
class Correction:
    reason: str
    message: str
    detail: str = ""


PUSH_UP_ASYM = Correction(
    "좌우 팔 비대칭",
    "양팔 균형을 맞춰 내려가세요",
    "왼팔과 오른팔의 굽힘 각도 차이가 큽니다. 양손에 균등하게 힘을 주세요.",
)
SQUAT_ASYM = Correction(
    "무릎 좌우 비대칭",
    "양발에 균등하게 체중을 실으세요",
    "왼쪽과 오른쪽 무릎 각도 차이가 큽니다. 골반과 무릎 균형을 맞춰 보세요.",
)
TRACKING_LOST = Correction(
    "관절 추적 끊김",
    "신체가 화면에 잘 보이게 움직이세요",
    "관절을 다시 인식할 때까지 횟수 집계를 기다립니다.",
)
RANGE_SHORT = "동작 범위 부족"
TARGET_HOLD_SHORT = "목표 각도 유지 부족"


@dataclass(frozen=True)
class Observation:
    angle: float
    left_angle: float = 0.0
    right_angle: float = 0.0
    fault: Correction | None = None


@dataclass(frozen=True)
class RepUpdate:
    result: bool | None
    feedback: str
    correction: Correction | None = None
    interrupted: bool = False


@dataclass(frozen=True)
class VisionResult:
    exercise: str
    phase: str                  # "align" | "countdown" | "exercise"
    aligned: bool
    countdown_left: int
    reps: int
    judged_attempts: int
    accuracy: float
    angle: float | None
    left_angle: float
    right_angle: float
    feedback: str
    feedback_detail: str
    is_correction: bool
    rep_completed: bool
    rejections: dict[str, int]