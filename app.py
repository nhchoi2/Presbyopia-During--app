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
from utils.face_detect import detect_faces, estimate_age  # 🔹 analyze_skin 제거
from utils.sidebar import load_sidebar  # ✅ 사이드바 추가

# 모델 및 라벨 로드
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

# 🔹 사이드바 로드
mode, theme = load_sidebar()  # ✅ 사이드바 추가

# 🔹 테마 적용
if theme == "다크 모드":
    st.set_page_config(page_title="동안 vs 노안 판별기", layout="wide", initial_sidebar_state="expanded", theme={"primaryColor": "#1f77b4"})
elif theme == "라이트 모드":
    st.set_page_config(page_title="동안 vs 노안 판별기", layout="wide", initial_sidebar_state="expanded", theme={"primaryColor": "#ff7f0e"})

st.title("📷 동안 vs 노안 판별기")
st.info("사진을 업로드하면 AI가 동안인지 노안인지 판별해줍니다.")

if mode == "개별 분석":
    st.header("📸 **개별 분석**")
    uploaded_file = st.file_uploader("📷 사진을 업로드하세요.", type=["jpg", "png", "jpeg", "webp"])

    if uploaded_file:
        image = Image.open(uploaded_file)

        # 🔹 업로드한 이미지 표시
        with st.container():
            st.image(image, caption="업로드한 이미지", width=300, use_container_width=True)

        # 🔹 모델 예측 수행
        size = (224, 224)
        image_resized = image.resize(size)
        image_array = np.asarray(image_resized).astype(np.float32) / 127.5 - 1
        data = np.expand_dims(image_array, axis=0)

        prediction = model.predict(data)
        index = np.argmax(prediction)
        result_label = class_names[index]
        confidence_score = prediction[0][index]

        # 🔹 결과 출력
        st.success(f"📌 **AI 판별 결과: {result_label[2:]}**")
        st.write(f"확신도: **{confidence_score:.2%}**")

        # 🔹 피드백 메시지
        feedback_message = get_feedback(result_label.strip())
        st.markdown(f"💬 **피드백:** {feedback_message}")


elif mode == "친구와 비교":
    st.header("👬 **친구와 동안 점수 비교**")
    uploaded_file_1 = st.file_uploader("📷 첫 번째 사진을 업로드하세요", type=["jpg", "png", "jpeg"], key="file1")
    uploaded_file_2 = st.file_uploader("📷 두 번째 사진을 업로드하세요", type=["jpg", "png", "jpeg"], key="file2")

    if uploaded_file_1 and uploaded_file_2:
        image_1 = Image.open(uploaded_file_1)
        image_2 = Image.open(uploaded_file_2)

        # 🔹 업로드된 두 이미지를 나란히 표시
        with st.container():
            st.image([image_1, image_2], caption=["첫 번째 사진", "두 번째 사진"], width=250)

        # 🔹 동안 점수 계산 함수
        def get_young_score(image):
            try:
                image_resized = image.resize((224, 224))
                image_array = np.asarray(image_resized).astype(np.float32) / 127.5 - 1
                data = np.expand_dims(image_array, axis=0)

                prediction = model.predict(data)
                confidence_score = prediction[0][0]
                return max(confidence_score * 100, 42)
            except Exception:
                return 42  # 기본 값 반환

        score_1 = get_young_score(image_1)
        score_2 = get_young_score(image_2)

        # 🔹 동안 점수 비교 결과
        st.subheader("📊 동안 점수 비교 결과")
        st.write(f"🔹 첫 번째 사진: **{score_1:.1f}/100**")
        st.write(f"🔹 두 번째 사진: **{score_2:.1f}/100**")

        if score_1 > score_2:
            st.success("🎉 **첫 번째 사진이 더 동안입니다!**")
        elif score_1 < score_2:
            st.success("🎉 **두 번째 사진이 더 동안입니다!**")
        else:
            st.info("🤝 **두 사람의 동안 점수가 같습니다!**")
