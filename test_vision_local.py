import os

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
from vision import Exercise, VisionEngine, open_available_camera


def draw_test_hud(frame, res):
    """로컬 단독 테스트용 상태 HUD(영문/숫자)를 좌측 상단에 표시합니다."""
    cv2.rectangle(frame, (15, 15), (430, 145), (25, 25, 25), -1)

    phase_text = f"{res.phase.upper()}"
    if res.phase == "countdown":
        phase_text += f" ({res.countdown_left}s)"

    cv2.putText(
        frame,
        f"PHASE: {phase_text}",
        (25, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"REPS: {res.reps} / ATTEMPTS: {res.judged_attempts} ({res.accuracy}%)",
        (25, 74),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (117, 238, 146),
        2,
    )
    ang_str = f"{res.angle:.1f} deg" if res.angle is not None else "---"
    cv2.putText(
        frame,
        f"ANGLE: {ang_str}  (L:{res.left_angle:.0f} / R:{res.right_angle:.0f})",
        (25, 106),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 225, 255),
        2,
    )
    cv2.putText(
        frame,
        "KEYS: [1]Squat  [2]Pushup  [3]Situp  [R]Reset  [Q]Quit",
        (25, 134),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (190, 190, 190),
        1,
    )


def main():
    engine = VisionEngine(skip_align=False)
    engine.reset(Exercise.SQUAT)

    cap = open_available_camera(preferred_index=0)
    print("[INFO] 웹캠 실행 중... (1: 스쿼트 정면 / 2: 팔굽혀펴기 정면 / 3: 윗몸일으키기 오른쪽 측면 / Q: 종료)")

    try:
        while True:
            ret, raw_frame = cap.read()
            if not ret or raw_frame is None:
                continue

            annotated, res = engine.process_frame(raw_frame, mirror=True)
            draw_test_hud(annotated, res)

            tag = "[교정]" if res.is_correction else "[코칭]"
            print(
                f"\r{tag} ({res.exercise}/{res.phase}) {res.feedback} | "
                f"성공: {res.reps}회 | 불인정: {res.rejections}     ",
                end="",
            )
            cv2.imshow("AI PT Trainer - Vision Only Test", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("1"):
                engine.reset(Exercise.SQUAT)
            elif key == ord("2"):
                engine.reset(Exercise.PUSH_UP)
            elif key == ord("3"):
                engine.reset(Exercise.SIT_UP)
            elif key == ord("r"):
                engine.reset()
    finally:
        print("\n[INFO] 종료합니다.")
        cap.release()
        cv2.destroyAllWindows()
        engine.close()


if __name__ == "__main__":
    main()

# Gradio UI, Pandas/Matplotlib에 전달할 호출 규격
# from vision import Exercise, VisionEngine

# engine = VisionEngine()
# engine.reset(Exercise.PUSH_UP)                         # 종목 변경 및 카운트 초기화
# annotated_bgr, result = engine.process_frame(frame)    # result.reps, result.accuracy, result.feedback, result.rejections