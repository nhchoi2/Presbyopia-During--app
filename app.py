import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# utils 모듈 임포트
from utils.face_detect import detect_and_crop_face
from utils.feedback import get_feedback
from utils.share_link import get_share_links
from utils.ai_chatbot import get_ai_response

def load_model(model_path):
    """
    keras_model.h5 파일을 로드해 반환합니다.
    """
    model = tf.keras.models.load_model(model_path)
    return model

def load_labels(label_path):
    """
    labels.txt 파일을 읽어 라벨 리스트를 반환합니다.
    """
    with open(label_path, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f.readlines()]
    return labels

def main():
    st.title("동안 vs 노안 판별기 👶🧓")

    # 모델 및 라벨 불러오기
    model_path = "model/keras_model.h5"
    label_path = "model/labels.txt"

    model = load_model(model_path)
    labels = load_labels(label_path)

    st.write("Google Teachable Machine으로 학습된 모델을 사용합니다.")
    st.write("사진을 업로드하면, 모델이 동안인지 노안인지 판별합니다.")

    # 사진 업로드
    uploaded_file = st.file_uploader("사진을 업로드하세요", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # PIL 이미지 열기
        image = Image.open(uploaded_file)

        # 얼굴 자동 크롭(옵션)
        cropped_image = detect_and_crop_face(image)

        # 이미지 표시
        st.image(cropped_image, caption="업로드한 이미지(자동 얼굴 크롭 적용)", use_column_width=True)

        # 모델 입력을 위한 전처리
        # Google Teachable Machine 기본 모델 입력 크기(224x224) 등으로 가정
        processed_img = cropped_image.resize((224, 224))
        img_array = np.array(processed_img) / 255.0  # 스케일링
        img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)

        # 예측
        predictions = model.predict(img_array)
        # 예: [ [0.2, 0.8] ] 형태라고 가정
        pred_index = np.argmax(predictions[0])
        confidence = predictions[0][pred_index]

        result_label = labels[pred_index]

        # 결과 출력
        st.markdown(f"**판정 결과:** {result_label}")
        st.markdown(f"**확률:** {confidence * 100:.2f}%")

        # 피드백 메시지
        feedback_msg = get_feedback(result_label)
        st.info(feedback_msg)

        # SNS 공유 링크
        share_links = get_share_links(result_label)
        st.markdown("**결과를 공유해보세요!**")
        st.markdown(f"[트위터로 공유하기]({share_links['twitter']})")
        st.markdown(f"[페이스북으로 공유하기]({share_links['facebook']})")

    st.markdown("---")
    st.header("AI 챗봇과 대화하기")
    user_input = st.text_input("챗봇에게 궁금한 것을 물어보세요.")
    if user_input:
        chatbot_answer = get_ai_response(user_input)
        st.write(f"**AI 답변:** {chatbot_answer}")

if __name__ == "__main__":
    main()
