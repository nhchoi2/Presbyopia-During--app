import os
from dotenv import load_dotenv

# 현재 디렉토리에서 .env 파일 강제 로드
dotenv_path = os.path.join(os.getcwd(), ".env")

if os.path.exists(dotenv_path):
    print(f"🔍 .env 파일 찾음: {dotenv_path}")
    load_dotenv(dotenv_path)  # 강제 로드
else:
    print("❌ .env 파일이 존재하지 않습니다. 파일 위치를 확인하세요.")

# 환경 변수 가져오기
API_KEY = os.getenv("OPENAI_API_KEY")

# API 키 확인 로그 추가
if API_KEY:
    print(f"✅ OpenAI API Key Loaded: {API_KEY[:10]}********")
else:
    print("❌ OpenAI API 키 없음! .env 파일을 확인하세요.")
