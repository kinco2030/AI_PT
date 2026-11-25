import numpy as np


def calculate_angle(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
) -> float:
    """
    점 b를 꼭짓점으로 하는 세 관절 좌표 (a, b, c) 사이의 각도(0~180도)를 계산합니다.
    """
    a_arr = np.array(a, dtype=np.float64)
    b_arr = np.array(b, dtype=np.float64)
    c_arr = np.array(c, dtype=np.float64)

    radians = np.arctan2(c_arr[1] - b_arr[1], c_arr[0] - b_arr[0]) - np.arctan2(
        a_arr[1] - b_arr[1], a_arr[0] - b_arr[0]
    )
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360.0 - angle
    return float(angle)


def _get_vis(landmarks, idx: int) -> float:
    vis = landmarks[idx].visibility
    return float(vis) if vis is not None else 1.0


def check_posture_alignment(
    landmarks,
    exercise: str,
    visibility_threshold: float = 0.45,
    is_mirrored: bool = True,
) -> tuple[str | None, bool, str]:
    """
    운동 종목별 카메라 위치 및 정렬 상태를 검증합니다.
    - pushup: 정면(FRONT) - 양쪽 어깨·팔꿈치·손목 확인
    - squat : 정면(FRONT) - 양쪽 어깨·엉덩이·무릎·발목 확인
    - situp : 오른쪽 측면(RIGHT-SIDE) - 웹캠이 사용자의 오른쪽에 위치했는지 확인
    """
    if exercise == "pushup":
        upper_indices = [11, 12, 13, 14, 15, 16]
        if min(_get_vis(landmarks, i) for i in upper_indices) < visibility_threshold:
            return None, False, "양쪽 어깨, 팔꿈치, 손목이 모두 보이도록 화면 안으로 들어와 주세요."

        shoulder_dx = abs(landmarks[11].x - landmarks[12].x)
        shoulder_dy = abs(landmarks[11].y - landmarks[12].y)

        if shoulder_dx < 0.14:
            return "FRONT", False, "웹캠을 정면으로 바라보고 양어깨가 넓게 보이도록 자세를 잡아주세요."
        if shoulder_dy > 0.15:
            return "FRONT", False, "양쪽 어깨의 높이를 수평으로 맞춰 주세요."

        return "FRONT", True, "정면 정렬 완료 (양팔 어깨-팔꿈치-손목 각도 측정 중)"

    if exercise == "squat":
        full_body_indices = [11, 12, 23, 24, 25, 26, 27, 28]
        if min(_get_vis(landmarks, i) for i in full_body_indices) < visibility_threshold:
            return None, False, "어깨부터 발목까지 전신이 화면에 나오도록 뒤로 물러서 주세요."

        shoulder_dx = abs(landmarks[11].x - landmarks[12].x)
        hip_dx = abs(landmarks[23].x - landmarks[24].x)

        if shoulder_dx < 0.14 or hip_dx < 0.08:
            return "FRONT", False, "웹캠을 정면으로 바라보고 양발을 어깨너비로 벌려 서주세요."

        return "FRONT", True, "정면 정렬 완료 (양쪽 무릎 각도 측정 중)"

    # situp: 발목 제외, 어깨(11,12)-엉덩이(23,24)-무릎(25,26)만 확인
    left_vis = float(np.mean([_get_vis(landmarks, i) for i in (11, 23, 25)]))
    right_vis = float(np.mean([_get_vis(landmarks, i) for i in (12, 24, 26)]))

    if max(left_vis, right_vis) < visibility_threshold:
        return None, False, "어깨, 엉덩이, 무릎이 모두 보이도록 카메라와 거리를 조절해 주세요."

    # 좌우 반전(거울 모드) 여부에 관계없이 카메라에 더 잘 보이는 쪽 관절 인덱스 선택
    camera_facing_side = "LEFT" if left_vis >= right_vis else "RIGHT"
    s_idx, h_idx, k_idx = (11, 23, 25) if camera_facing_side == "LEFT" else (12, 24, 26)

    # 웹캠이 내 오른쪽에 있을 때:
    # - 거울 모드(is_mirrored=True): 머리가 화면 오른쪽, 무릎이 왼쪽(knee.x < hip.x)
    # - 원본 모드(is_mirrored=False): 머리가 화면 왼쪽, 무릎이 오른쪽(knee.x > hip.x)
    hip_x = landmarks[h_idx].x
    knee_x = landmarks[k_idx].x
    is_webcam_on_my_right = (knee_x < hip_x) if is_mirrored else (knee_x > hip_x)

    if not is_webcam_on_my_right:
        return camera_facing_side, False, "웹캠이 내 오른쪽에 오도록 반대 방향으로 누워주세요."

    # 몸이 가로(누운 자세)로 배치되었는지 확인
    body_dx = abs(landmarks[s_idx].x - hip_x)
    shoulder_overlap_dx = abs(landmarks[11].x - landmarks[12].x)
    if body_dx < 0.08 and shoulder_overlap_dx > 0.25:
        return camera_facing_side, False, "오른쪽 옆모습이 보이도록 웹캠과 수직 방향으로 누워주세요."

    return camera_facing_side, True, "오른쪽 측면 정렬 완료 (고관절 각도 측정 중)"