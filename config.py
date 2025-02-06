import os
from dotenv import load_dotenv
import streamlit as st

# .env 파일 로드 (로컬 실행용)
dotenv_path = os.path.join(os.getcwd(), ".env")
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)

# 환경 변수 가져오기 (Streamlit Secrets도 함께 확인)
API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
DEBUG_MODE = os.getenv("DEBUG", "False").lower() == "true"

# API 키 확인 로그 추가
if API_KEY:
    print(f"✅ OpenAI API Key Loaded: {API_KEY[:10]}********")
else:
    print("❌ OpenAI API 키 없음! .env 또는 Streamlit Secrets를 확인하세요.")
