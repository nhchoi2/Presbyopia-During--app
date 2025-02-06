import random

def get_feedback(label: str) -> str:
    """
    모델 판별 결과 '동안'/'노안'에 따라
    랜덤 피드백 메시지를 반환합니다.
    """
    young_messages = ['🎉 완벽한 동안! 신분증 검사 필수네요!',
                      '🧒 어릴 때부터 지금까지 똑같은 얼굴?!',
                      '✨ 동안 유지 비법 좀 알려주세요!',
                      '📸 초등학생 때 찍은 사진이랑 똑같은데요?',
                      '⏳ 시간이 멈춘 얼굴! 비결이 뭔가요?',
                      '🧴 피부관리 따로 하세요? 너무 탱탱해요!',
                      '🏆 동안계의 챔피언 등극! 축하드립니다!',
                      '🔥 나이가 숫자라는 걸 몸소 증명하는 중!',
                      '🎭 학생 할인도 가능할 듯한 동안 미모!',
                      '🥇 동안 클래스의 정점! 동안 끝판왕!',
                      ]
    old_messages = ['🏆 노안이지만 멋있어요! 신뢰감 폭발!',
                    '🔥 세월의 흔적이 느껴지는 얼굴… 하지만 카리스마는 최고!',
                    '🎩 인생의 깊이가 보이는 얼굴! 중후한 매력 발산 중!',
                    '💼 지적인 분위기! 노안이 아니라 성숙미라고 해야죠!',
                    '🍷 노안은 곧 클래식한 매력! 중후한 멋이 있네요!',
                    '🔍 노안이지만 대기업 임원 느낌이 나는 얼굴입니다!',
                    '🧑‍⚖ 믿음직한 인상! 어릴 때도 어른 같았다는 말 많이 들으셨죠?',
                    '🏅 노안이지만 동안보다 더 멋진 분위기! 분위기 깡패!',
                    '🕶 노안이라기엔 너무 세련된 분위기! 모델 포스 납니다!',
                    '🤵 노안이지만 나이보다 더 성숙하고 댄디한 이미지!',
    ]
    
    if label == "동안":
        return random.choice(young_messages)
    else:
        return random.choice(old_messages)
