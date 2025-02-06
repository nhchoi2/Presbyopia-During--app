import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
from config import API_KEY, DEBUG_MODE
import openai

# OpenAI API 키 설정
if not API_KEY:
    raise ValueError("❌ OpenAI API 키가 설정되지 않았습니다.")
openai.api_key = API_KEY

# 디버그 모드 체크
if DEBUG_MODE:
    st.info("🔧 디버그 모드 활성화 중입니다.")

# AI 모델 로드
MODEL_PATH = "model/keras_model.h5"
LABELS_PATH = "model/labels.txt"

@st.cache_resource
def load_ai_model():
    return load_model(MODEL_PATH, compile=False)

@st.cache_resource
def load_labels():
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]

model = load_ai_model()
labels = load_labels()

# Streamlit UI
st.title("🔍 동안 vs 노안 판별기")
st.info("사진을 업로드하면 AI가 분석하여 동안인지 노안인지 판별해드립니다!")

# 파일 업로드
uploaded_file = st.file_uploader("사진을 업로드하세요", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # 업로드된 이미지 표시
    image = Image.open(uploaded_file)
    st.image(image, caption="업로드한 이미지", use_column_width=True)

    # 이미지 전처리
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image) / 127.5 - 1
    img = np.expand_dims(image_array, axis=0)

    # AI 모델로 예측
    prediction = model.predict(img)
    result_idx = np.argmax(prediction)
    result_label = labels[result_idx]
    confidence_score = prediction[0][result_idx]

    # 결과 출력
    st.subheader("📌 AI 판별 결과")
    st.info(f"**이 얼굴은 {result_label}입니다.** 확률: {confidence_score:.2%}")

    # 결과에 따른 피드백
    feedback = {
        "동안": [
            "완벽한 동안! 신분증 검사 걱정 없겠네요!",
            "어릴 때부터 지금까지 똑같은 얼굴?!",
            "동안 유지 비법 좀 알려주세요!",
            "초등학생 때도 이 얼굴이었을 듯?",
            "어디서 시간을 멈추셨나요?"
        ],
        "노안": [
            "노안이지만 멋있어요! 신뢰감 폭발!",
            "세월의 흔적이 느껴지는 얼굴… 하지만 카리스마는 최고!",
            "인생의 깊이가 보이는 얼굴!",
            "멋진 중후한 매력이 느껴져요!",
            "노안이라고요? 그냥 어른스러운 거죠!"
        ]
    }

    random_feedback = np.random.choice(feedback[result_label])
    st.write(f"💬 **AI 피드백:** {random_feedback}")

    # AI 챗봇 질문
    st.subheader("🤖 동안/노안 관련 AI 상담")
    question = st.text_input("동안/노안 관련 질문을 입력하세요!")
    if question:
        with st.spinner("AI가 답변을 작성 중입니다..."):
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": question}]
                )
                answer = response["choices"][0]["message"]["content"]
                st.write(f"🤖 AI 답변: {answer}")
            except Exception as e:
                st.error(f"❌ OpenAI API 호출 중 오류 발생: {str(e)}")

