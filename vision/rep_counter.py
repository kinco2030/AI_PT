from dataclasses import replace
import math

from vision.rules import (
    Correction,
    Exercise,
    FAULT_HOLD,
    MAX_FRAME_GAP,
    Observation,
    POSITION_HOLD,
    RANGE_SHORT,
    RULES,
    RepUpdate,
    TARGET_HOLD_SHORT,
    TRACKING_LOST,
)


class RepCounter:
    """
    단순 각도 통과가 아닌 자세 유지 시간(POSITION_HOLD), 최대 굽힘 각도(best_angle),
    지속적 자세 결함(FAULT_HOLD)을 종합해 1회 성공/불인정을 판정하는 상태 머신입니다.
    """

    def __init__(self, exercise: Exercise):
        self.rule = RULES[exercise]
        self.reset()

    def reset(self):
        self.phase = "seek_start"
        self.last_seen: float | None = None
        self.zone_since: float | None = None
        self.zone: str | None = None
        self.fault_since: float | None = None
        self.fault_seen: Correction | None = None
        self.rep_fault: Correction | None = None
        self.best_angle: float | None = None

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
        reached = self.best_angle is not None and self.best_angle <= rule.target
        if reached:
            return Correction(
                TARGET_HOLD_SHORT,
                "목표 자세를 잠깐 유지하세요",
                f"{rule.target:g}° 이하 구간을 약 {POSITION_HOLD:g}초 유지해야 합니다.",
            )
        best_val = self.best_angle if self.best_angle is not None else rule.start
        return Correction(
            RANGE_SHORT,
            rule.shallow,
            f"이번 최저 {best_val:.1f}° · 목표 {rule.target:g}° 이하",
        )

    def _follow(self, observation: Observation | None, now: float) -> RepUpdate:
        if observation is None or not math.isfinite(observation.angle):
            self.zone_since = self.fault_since = None
            self.fault_seen = None
            return RepUpdate(None, TRACKING_LOST.message, TRACKING_LOST)

        self.last_seen = now
        rule = self.rule
        angle = observation.angle

        at_start = angle >= rule.start
        departed = angle <= rule.departure
        at_target = angle <= rule.target

        if self.phase == "seek_start":
            if self._held(at_start, now):
                self._move_to("ready")
                return RepUpdate(None, rule.cue)
            return RepUpdate(None, f"시작 자세({rule.start:g}° 이상)를 잠시 유지하세요.")

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
        if observation.fault and self.fault_since is not None and now - self.fault_since >= FAULT_HOLD:
            correction = observation.fault
            self.rep_fault = self.rep_fault or correction

        if self.phase == "descending":
            if self.best_angle is None:
                self.best_angle = angle
            self.best_angle = min(self.best_angle, angle)

            zone = "target" if at_target else "start" if at_start else None
            if zone != self.zone:
                self.zone_since = None
                self.zone = zone
            held = self._held(zone is not None, now)

            if held and zone == "target":
                if rule.count_at_target:
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