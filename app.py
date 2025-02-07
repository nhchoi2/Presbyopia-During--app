import sys
import os
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from utils.feedback import get_feedback
from utils.sidebar import load_sidebar

# 모델 및 라벨 로드
MODEL_PATH = "model/keras_model.h5"
LABELS_PATH = "model/labels.txt"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

@st.cache_resource
def load_labels():
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]

model = load_model()
class_names = load_labels()

# 🔹 사이드바 로드
mode, theme = load_sidebar()

# 페이지 설정
st.set_page_config(page_title="동안 vs 노안 판별기", layout="wide")

# 헤더 섹션
with st.container():
    st.title("동안 vs 노안 판별기")
    st.subheader("AI가 당신의 얼굴을 분석하여 동안인지 노안인지 알려드립니다.")
    st.markdown("#### 📸 사진을 업로드하고 결과를 확인하세요!")

# 분석 모드 선택
st.markdown("### 분석 모드")
col1, col2 = st.columns(2)

with col1:
    if st.button("개별 분석"):
        mode = "개별 분석"
with col2:
    if st.button("친구와 비교"):
        mode = "친구와 비교"

# 분석 섹션
if mode == "개별 분석":
    st.markdown("### 📷 개별 분석")
    uploaded_file = st.file_uploader("사진을 업로드하세요.", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="업로드한 사진", use_column_width=True)

        # 예측 수행
        size = (224, 224)
        image_resized = image.resize(size)
        image_array = np.asarray(image_resized).astype(np.float32) / 127.5 - 1
        data = np.expand_dims(image_array, axis=0)

        prediction = model.predict(data)
        index = np.argmax(prediction)
        result_label = class_names[index]
        confidence_score = prediction[0][index]

        st.success(f"결과: {result_label[2:]}")
        st.write(f"확신도: {confidence_score:.2%}")
        st.markdown(f"💬 피드백: {get_feedback(result_label.strip())}")

elif mode == "친구와 비교":
    st.markdown("### 👬 친구와 비교")
    uploaded_file_1 = st.file_uploader("첫 번째 사진을 업로드하세요", type=["jpg", "png", "jpeg"], key="file1")
    uploaded_file_2 = st.file_uploader("두 번째 사진을 업로드하세요", type=["jpg", "png", "jpeg"], key="file2")

    if uploaded_file_1 and uploaded_file_2:
        image_1 = Image.open(uploaded_file_1)
        image_2 = Image.open(uploaded_file_2)

        st.image([image_1, image_2], caption=["첫 번째 사진", "두 번째 사진"], width=300)

        def get_young_score(image):
            try:
                image_resized = image.resize((224, 224))
                image_array = np.asarray(image_resized).astype(np.float32) / 127.5 - 1
                data = np.expand_dims(image_array, axis=0)

                prediction = model.predict(data)
                confidence_score = prediction[0][0]
                return max(confidence_score * 100, 42)
            except Exception:
                return 42

        score_1 = get_young_score(image_1)
        score_2 = get_young_score(image_2)

        st.write(f"첫 번째 사진 동안 점수: {score_1:.1f}/100")
        st.write(f"두 번째 사진 동안 점수: {score_2:.1f}/100")
        if score_1 > score_2:
            st.success("첫 번째 사진이 더 동안입니다!")
        elif score_1 < score_2:
            st.success("두 번째 사진이 더 동안입니다!")
        else:
            st.info("두 사진의 동안 점수가 같습니다!")
