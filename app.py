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
from utils.face_detect import detect_faces, estimate_age, analyze_skin  # ✅ 얼굴 분석 기능

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

# 🔹 "개별 분석" 또는 "비교 분석" 선택
mode = st.radio("🔍 분석 모드 선택", ["개별 분석", "친구와 비교"])

if mode == "개별 분석":
    # 📌 기존 개별 분석 기능
    uploaded_file = st.file_uploader("📷 사진을 업로드하세요.", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="업로드한 이미지", use_column_width=True)

        # 모델 입력 형태로 변환
        size = (224, 224)
        image_resized = image.resize(size)
        image_array = np.asarray(image_resized).astype(np.float32) / 127.5 - 1
        data = np.expand_dims(image_array, axis=0)

        # AI 예측 수행
        prediction = model.predict(data)
        index = np.argmax(prediction)
        result_label = class_names[index]
        confidence_score = prediction[0][index]

        # 결과 출력
        st.subheader(f"📌 AI 판별 결과: {result_label}")
        st.write(f"확신도: {confidence_score:.2%}")

        # 🔹 얼굴 검출 추가
        faces = detect_faces(image)
        if isinstance(faces, list) and len(faces) > 0:
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

elif mode == "친구와 비교":
    # 📌 친구와 비교 기능
    st.subheader("👬 친구와 동안 점수 비교")
    
    uploaded_file_1 = st.file_uploader("📷 첫 번째 사진을 업로드하세요", type=["jpg", "png", "jpeg"], key="file1")
    uploaded_file_2 = st.file_uploader("📷 두 번째 사진을 업로드하세요", type=["jpg", "png", "jpeg"], key="file2")

    if uploaded_file_1 and uploaded_file_2:
        image_1 = Image.open(uploaded_file_1)
        image_2 = Image.open(uploaded_file_2)

        st.image([image_1, image_2], caption=["첫 번째 사진", "두 번째 사진"], width=250)

        # 🔹 동안 점수 계산 함수
        def get_young_score(image):
            image_resized = image.resize((224, 224))
            image_array = np.asarray(image_resized).astype(np.float32) / 127.5 - 1
            data = np.expand_dims(image_array, axis=0)

            prediction = model.predict(data)
            confidence_score = prediction[0][np.argmax(prediction)]  # 🔹 동안일 확률
            return confidence_score * 100  # 100점 만점 변환

        # 동안 점수 계산
        score_1 = get_young_score(image_1)
        score_2 = get_young_score(image_2)

        # 🔹 결과 비교 출력
        st.subheader("📊 동안 점수 비교")
        st.write(f"🔹 첫 번째 사진 동안 점수: **{score_1:.1f}/100**")
        st.write(f"🔹 두 번째 사진 동안 점수: **{score_2:.1f}/100**")

        # 🔹 비교 결과 출력
        if score_1 > score_2:
            st.success("🎉 **첫 번째 사진이 더 동안입니다!**")
        elif score_1 < score_2:
            st.success("🎉 **두 번째 사진이 더 동안입니다!**")
        else:
            st.info("🤝 **두 사람의 동안 점수가 같습니다!**")

