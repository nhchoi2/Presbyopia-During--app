import random

def get_feedback(label: str) -> str:
    """
    모델 판별 결과 '동안'/'노안'에 따라
    랜덤 피드백 메시지를 반환합니다.
    """
    young_messages = [
        "정말 동안이시네요!",
        "피부가 탱탱하고 건강해 보여요!",
        "멋진 동안 미모를 유지하세요!"
    ]
    old_messages = [
        "중후함이 멋스러우시네요!",
        "깊이 있는 매력이 느껴집니다!",
        "연륜이 돋보이는 멋진 모습이세요!"
    ]
    
    if label == "동안":
        return random.choice(young_messages)
    else:
        return random.choice(old_messages)
