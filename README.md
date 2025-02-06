# Presbyopia-During--app
# 프로젝트 구조조
# /your_project
# │── app.py                  # 메인 실행 파일 (Streamlit 앱)
# │── config.py               # 환경 변수 로드
# │── requirements.txt        # 필수 라이브러리 목록
# │── .env                    # 환경 변수 저장 (OpenAI API 키 등)
# │
# ├── /model                  # AI 모델 관련 파일
# │   ├── keras_model.h5      # Google Teachable Machine 모델 파일
# │   └── labels.txt          # 모델 클래스 라벨 파일
# │
# └── /utils                  # 추가 기능 모듈
#     ├── face_detect.py      # 얼굴 감지 및 자동 크롭
#     ├── feedback.py         # 동안/노안 피드백 제공
#     ├── share_link.py       # SNS 공유 링크 생성
#     └── ai_chatbot.py       # AI 챗봇 (OpenAI API 활용)
