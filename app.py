import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 모델 및 라벨 파일 경로
MODEL_PATH = "model/keras_model.h5"
LABELS_PATH = "model/labels.txt"

# 모델 로드
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

# 라벨 로드
def load_labels():
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f.readlines()]
    return labels

# 모델 및 라벨 불러오기
model = load_model()
labels = load_labels()

# Streamlit UI
st.title("🔍 동안 vs 노안 판별기")
st.write("사진을 업로드하면 AI가 분석하여 동안인지 노안인지 판별해드립니다!")

st.markdown("""
### 📌 사용 방법
1. 사진을 업로드하세요.
2. AI가 분석하여 동안/노안을 판별합니다.
3. 결과를 확인하고 재미있게 활용해보세요! 🎉
""")

# 파일 업로드 기능
uploaded_file = st.file_uploader("사진을 업로드하세요", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # 이미지 표시
    image = Image.open(uploaded_file)
    st.image(image, caption="업로드한 이미지", use_column_width=True)

    # 이미지 전처리
    img = image.resize((224, 224))  # Teachable Machine 기본 입력 크기
    img = np.array(img) / 255.0  # 정규화
    img = np.expand_dims(img, axis=0)  # 배치 차원 추가

    # 예측 수행
    prediction = model.predict(img)
    result_idx = np.argmax(prediction)  # 가장 높은 확률의 인덱스
    result_label = labels[result_idx]  # 해당 인덱스의 라벨

    # 결과 출력
    st.subheader("📌 결과 분석")
    st.write(f"**🔍 AI 판별 결과:** {result_label}")
    st.write(f"📊 확률: {prediction[0][result_idx] * 100:.2f}%")

    # 동안/노안에 따른 추가 메시지
    if result_label == "동안":
        st.success("🎉 축하합니다! AI가 동안으로 판별했습니다! 😍")
    else:
        st.warning("🤔 AI가 노안으로 판별했어요. 하지만 자신감을 가지세요! 💪")

