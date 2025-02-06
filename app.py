import streamlit as st
from keras.models import load_model  # TensorFlow가 필요
from PIL import Image, ImageOps  # Pillow 설치 필요
import numpy as np

# Streamlit 앱 메인 함수
def main():
    st.title("🔍 동안 vs 노안 판별기")
    st.info("사진을 업로드하면 AI가 분석하여 동안인지 노안인지 판별해드립니다!")

    # 이미지 업로드 기능
    image = st.file_uploader("이미지 파일을 업로드하세요.", type=["jpg", "png", "jpeg"])

    if image is not None:
        # 이미지 표시
        st.image(image, caption="업로드한 이미지", use_container_width=True)

        # 이미지 열기
        image = Image.open(image)

        # 모델 로드
        model = load_model("model/keras_model.h5", compile=False)

        # 라벨 로드
        class_names = open("model/labels.txt", "r", encoding="utf-8").readlines()

        # 데이터 배열 생성 (1개의 이미지, 224x224 크기, RGB 3채널)
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

        # 이미지 전처리 (224x224 크기로 변환 및 정규화)
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data[0] = normalized_image_array

        # 모델 예측 수행
        prediction = model.predict(data)
        index = np.argmax(prediction)  # 확률이 가장 높은 클래스 인덱스 선택
        class_name = class_names[index]  # 해당 인덱스의 라벨
        confidence_score = prediction[0][index]  # 신뢰도 점수

        # 결과 출력
        st.subheader("📌 AI 판별 결과")
        st.info(f"**이 얼굴은 {class_name[2:].strip()}입니다.** 확률: {confidence_score:.2%}")

        # 동안/노안에 따른 메시지 추가
        if "동안" in class_name:
            st.success("🎉 축하합니다! AI가 동안으로 판별했어요! 😍")
        else:
            st.warning("🤔 AI가 노안으로 판별했어요. 하지만 나이는 숫자일 뿐! 💪")

# 실행
if __name__ == "__main__":
    main()
