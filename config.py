import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 환경 변수 정의
API_KEY = os.getenv("OPENAI_API_KEY")
DEBUG_MODE = os.getenv("DEBUG", "False").lower() == "true"
