import cv2
import numpy as np
from PIL import Image
import streamlit as st
from deepface import DeepFace  # DeepFace 라이브러리 활용

st.title("🔍 얼굴 나이 예측")

uploaded_file = st.file_uploader("사진을 업로드하세요", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="업로드한 이미지", use_column_width=True)

    # OpenCV 얼굴 검출
    image_np = np.array(image)
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        st.write(f"✅ 검출된 얼굴 개수: {len(faces)}개")

        # 얼굴 부분만 크롭해서 분석
        x, y, w, h = faces[0]
        face_crop = image_np[y:y+h, x:x+w]

        # DeepFace를 이용한 나이 예측
        analysis = DeepFace.analyze(face_crop, actions=["age"], enforce_detection=False)
        estimated_age = analysis[0]["age"]

        st.subheader(f"📌 AI 예측 나이: {estimated_age}세")
    else:
        st.error("😔 얼굴을 감지하지 못했습니다. 더 밝은 환경에서 촬영해 주세요.")
