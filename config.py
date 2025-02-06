import os
from dotenv import load_dotenv
import streamlit as st

# .env 파일 로드
load_dotenv()

# 환경 변수 정의 (Streamlit Secrets도 함께 확인)
API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
DEBUG_MODE = os.getenv("DEBUG", "False").lower() == "true"

# API 키 확인 로그 추가
print(f"🔑 OpenAI API Key Loaded: {API_KEY[:10]}********" if API_KEY else "❌ OpenAI API 키 없음!")
