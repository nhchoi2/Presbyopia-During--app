import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import random
from utils.face_detect import crop_face
from utils.share_link import generate_share_link
from utils.ai_chatbot import ask_ai
from utils.feedback import get_random_feedback  # 랜덤 피드백 추가
from config import API_KEY

# API 키 확인
if API_KEY:
    print(f"✅ OpenAI API 키: {API_KEY}")
else:
    print("❌ API 키를 찾을 수 없습니다.")
    
# 모델 로드
model = load_model("model/keras_model.h5", compile=False)
labels = open("model/labels.txt", "r", encoding="utf-8").readlines()

# Streamlit UI
st.title("🔍 동안 vs 노안 판별기")
st.info("사진을 업로드하면 AI가 분석하여 동안인지 노안인지 판별해드립니다!")

# 파일 업로드 기능
uploaded_file = st.file_uploader("사진을 업로드하세요", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="업로드한 이미지", use_container_width=True)

    # 얼굴 감지 및 크롭
    image = crop_face(image)

    # 이미지 전처리
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image) / 127.5 - 1
    img = np.expand_dims(image_array, axis=0)

    # 예측 수행
    prediction = model.predict(img)
    result_idx = np.argmax(prediction)
    result_label = labels[result_idx].strip()
    confidence_score = prediction[0][result_idx]

    # 결과 출력
    st.subheader("📌 AI 판별 결과")
    st.info(f"**이 얼굴은 {result_label}입니다.** 확률: {confidence_score:.2%}")

    # 동안/노안에 따른 랜덤 피드백 추가
    feedback_text = get_random_feedback(result_label)
    st.write(f"💬 **AI 피드백:** {feedback_text}")

    # 동안/노안에 따른 메시지
    if "동안" in result_label:
        st.success("🎉 축하합니다! AI가 동안으로 판별했어요! 😍")
    else:
        st.warning("🤔 AI가 노안으로 판별했어요. 하지만 나이는 숫자일 뿐! 💪")

    # 결과 공유 링크 생성
    share_url = generate_share_link("https://your-app.com/static/result.jpg")
    st.write(f"📤 결과 공유: {share_url}")

# AI 챗봇 기능 추가
st.subheader("🤖 동안/노안 AI 상담")
question = st.text_input("동안/노안 관련 질문을 해보세요!")
if question:
    answer = ask_ai(question)
    st.write(f"🤖 AI 답변: {answer}")
