import random

def get_feedback(label: str) -> str:
    """
    모델 판별 결과 라벨(동안/노안 등)에 따라
    랜덤 피드백 메시지를 반환합니다.
    """

    young_messages = [
        "와, 정말 피부가 탱탱하네요!",
        "동안 미모가 빛이 납니다!",
        "자외선 차단을 정말 잘하셨나봐요!",
    ]

    old_messages = [
        "경륜이 느껴지는 멋진 모습이세요!",
        "인생의 깊이가 느껴집니다!",
        "중후함이 돋보이시네요!",
    ]

    if label == "동안":
        return random.choice(young_messages)
    else:
        # "노안"으로 가정
        return random.choice(old_messages)
