import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# 만약 feedback.py와 share_link.py를 사용하지 않는다면, 아래 import도 제거하세요.
from utils.feedback import get_feedback
from utils.share_link import get_share_links

def load_model(model_path: str):
    """
    Google Teachable Machine 등에서 export한 Keras 모델(keras_model.h5)을 로드합니다.
    """
    model = tf.keras.models.load_model(model_path)
    return model

def load_labels(label_path: str):
    """
    라벨 파일(labels.txt)을 읽어 라벨 리스트를 반환합니다.
    예: ['동안', '노안']
    """
    with open(label_path, 'r', encoding='utf-8') as f:
        labels = [line.strip() for line in f]
    return labels

def main():
    st.title("동안 vs 노안 판별기 👶🧓")
    st.write("사진을 업로드하면 AI가 동안인지 노안인지 판별합니다.")
    
    # 모델 및 라벨 파일 경로 설정
    model_path = "model/keras_model.h5"
    label_path = "model/labels.txt"
    
    # 모델과 라벨 불러오기
    model = load_model(model_path)
    labels = load_labels(label_path)

    # 업로드 파일 받기
    uploaded_file = st.file_uploader("사진을 업로드하세요", type=["jpg", "png", "jpeg","webp"])
    
    if uploaded_file is not None:
        # PIL 이미지 열기
        image = Image.open(uploaded_file)
        
        # 이미지 표시
        st.image(image, caption="업로드한 이미지", width=500)
        
        # 모델 입력 크기에 맞춰 리사이즈 (예: 224x224)
        resized_img = image.resize((224, 224))
        img_array = np.array(resized_img) / 255.0  # 0~1 스케일링
        img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)
        
        # 모델 예측
        predictions = model.predict(img_array)
        pred_index = np.argmax(predictions[0])
        confidence = predictions[0][pred_index]
        st.text(confidence)
        
        # 결과 라벨
        result_label = labels[pred_index]
        
        
        st.write(f"**결과 라벨:** {result_label[2:]}")
        st.write(f"**확률:** {confidence * 100:.2f}%")
        
        # 피드백 메시지 (필요 없다면 제거)
        feedback_msg = get_feedback(result_label[2:])
        st.info(feedback_msg)
        
        # SNS 공유 링크 (필요 없다면 제거)
        share_links = get_share_links(result_label)
        st.markdown("**결과를 공유해보세요!**")
        st.markdown(f"[트위터로 공유하기]({share_links['twitter']})")
        st.markdown(f"[페이스북으로 공유하기]({share_links['facebook']})")
    

if __name__ == "__main__":
    main()
