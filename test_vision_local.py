import cv2
from vision.pose_detector import PoseDetector
from vision.exercise_analyzer import ExerciseAnalyzer


def main():
    detector = PoseDetector()
    analyzer = ExerciseAnalyzer()
    analyzer.reset("situp")  # "squat", "pushup", "situp" 변경 테스트

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        detector.close()
        raise RuntimeError("웹캠을 열 수 없습니다.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            landmarks, active_side, is_aligned, guide_msg = detector.extract_landmarks(frame)

            if landmarks and active_side and is_aligned:
                status = analyzer.analyze(landmarks, active_side)
            else:
                status = {
                    "exercise": analyzer.exercise,
                    "count": analyzer.count,
                    "stage": analyzer.stage,
                    "angle": 0.0,
                    "feedback": guide_msg,
                    "accuracy": 0.0,
                }

            output_frame = detector.draw_overlay(
                frame_bgr=frame,
                landmarks=landmarks,
                is_aligned=is_aligned,
                exercise_name=status["exercise"],
                count=status["count"],
                stage=status["stage"],
                angle=status["angle"],
            )

            print(
                f"\r[상태] {status['feedback']} | 횟수: {status['count']} | 각도: {status['angle']}",
                end="",
            )
            cv2.imshow("AI PT Trainer - Vision Module Test", output_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("1"):
                analyzer.reset("squat")
            elif key == ord("2"):
                analyzer.reset("pushup")
            elif key == ord("3"):
                analyzer.reset("situp")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()


if __name__ == "__main__":
    main()