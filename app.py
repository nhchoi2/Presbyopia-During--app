import sys
import os
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# 현재 디렉토리를 Python 경로에 추가하여 `utils` 폴더 인식 가능하게 설정
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from utils.feedback import get_feedback
from utils.share_link import get_share_links
from utils.face_detect import detect_faces, estimate_age, analyze_skin  # ✅ 나이 예측 및 피부 분석 추가

# 모델 및 라벨 불러오기
MODEL_PATH = "model/keras_model.h5"
LABELS_PATH = "model/labels.txt"

@st.cache_resource
def load_model():
    """저장된 모델을 불러오는 함수"""
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

@st.cache_resource
def load_labels():
    """라벨 파일을 불러오는 함수"""
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]

# 모델과 라벨 로드
model = load_model()
class_names = load_labels()

# Streamlit UI 설정
st.title("📷 동안 vs 노안 판별기")
st.info("사진을 업로드하면 AI가 동안인지 노안인지 판별해줍니다.")

# 파일 업로드 기능
uploaded_file = st.file_uploader("사진을 업로드하세요.", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="업로드한 이미지", use_column_width=True)

    # 모델 입력 형태로 변환
    size = (224, 224)
    image_resized = image.resize(size)
    image_array = np.asarray(image_resized).astype(np.float32) / 127.5 - 1
    data = np.expand_dims(image_array, axis=0)

    # 예측 수행
    prediction = model.predict(data)
    index = np.argmax(prediction)
    result_label = class_names[index]
    confidence_score = prediction[0][index]

    # 결과 출력
    st.subheader(f"📌 AI 판별 결과: {result_label}")
    st.write(f"확신도: {confidence_score:.2%}")

    # 🔹 얼굴 검출 추가 (오류 방지 처리 포함)
    faces = detect_faces(image)
    if isinstance(faces, list) and len(faces) > 0:  # ✅ 빈 리스트 처리 추가
        estimated_age = estimate_age(image)
        skin_result = analyze_skin(image)

        st.subheader(f"📌 AI 예측 나이: {estimated_age}세")  # 🔹 AI가 예측한 나이 출력
        st.write(skin_result)  # 🔹 피부 분석 결과 출력
    else:
        st.warning("😔 얼굴을 찾지 못했습니다. 더 밝은 환경에서 촬영해 주세요.")

    # 🔹 랜덤 피드백 제공
    feedback_message = get_feedback(result_label.strip())  # 🔹 라벨에서 공백 제거 후 피드백
    st.success(f"💬 {feedback_message}")

    # 🔹 SNS 공유 링크 추가
    st.subheader("🔗 결과 공유하기")
    share_links = get_share_links(result_label.strip())  # 🔹 라벨에서 공백 제거 후 링크 생성
    st.write(f"[트위터에서 공유하기]({share_links['twitter']})")
    st.write(f"[페이스북에서 공유하기]({share_links['facebook']})")
