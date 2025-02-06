import streamlit as st
from PIL import Image
from utils.model_utils import load_model, load_labels, predict_image
from utils.feedback import get_feedback
from utils.share_link import get_share_links
from utils.face_detect import detect_faces, estimate_age, analyze_skin
from utils.sidebar import load_sidebar  # ✅ 사이드바 추가

# 모델 및 라벨 불러오기
model = load_model()
class_names = load_labels()

# 🔹 사이드바 로드
mode, theme = load_sidebar()

st.title("📷 동안 vs 노안 판별기")
st.info("사진을 업로드하면 AI가 동안인지 노안인지 판별해줍니다.")

if mode == "개별 분석":
    uploaded_file = st.file_uploader("📷 사진을 업로드하세요.", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="업로드한 이미지", use_column_width=True)

        # AI 예측 수행
        result_label, confidence_score = predict_image(model, image, class_names)

        # 결과 출력
        st.subheader(f"📌 AI 판별 결과: {result_label}")
        st.write(f"확신도: {confidence_score:.2%}")

        # 🔹 얼굴 검출 추가
        faces = detect_faces(image)
        if faces:
            estimated_age = estimate_age(image)
            skin_result = analyze_skin(image)

            st.subheader(f"📌 AI 예측 나이: {estimated_age}세")
            st.write(skin_result)
        else:
            st.warning("😔 얼굴을 찾지 못했습니다. 더 밝은 환경에서 촬영해 주세요.")

        # 🔹 랜덤 피드백 제공
        feedback_message = get_feedback(result_label.strip())
        st.success(f"💬 {feedback_message}")

        # 🔹 SNS 공유 링크 추가
        st.subheader("🔗 결과 공유하기")
        share_links = get_share_links(result_label.strip())
        st.write(f"[트위터에서 공유하기]({share_links['twitter']})")
        st.write(f"[페이스북에서 공유하기]({share_links['facebook']})")

elif mode == "친구와 비교":
    st.subheader("👬 친구와 동안 점수 비교")

    uploaded_file_1 = st.file_uploader("📷 첫 번째 사진을 업로드하세요", type=["jpg", "png", "jpeg"], key="file1")
    uploaded_file_2 = st.file_uploader("📷 두 번째 사진을 업로드하세요", type=["jpg", "png", "jpeg"], key="file2")

    if uploaded_file_1 and uploaded_file_2:
        image_1 = Image.open(uploaded_file_1)
        image_2 = Image.open(uploaded_file_2)

        st.image([image_1, image_2], caption=["첫 번째 사진", "두 번째 사진"], width=250)

        score_1 = predict_image(model, image_1, class_names)[1] * 100
        score_2 = predict_image(model, image_2, class_names)[1] * 100

        st.subheader("📊 동안 점수 비교")
        st.write(f"🔹 첫 번째 사진 동안 점수: **{score_1:.1f}/100**")
        st.write(f"🔹 두 번째 사진 동안 점수: **{score_2:.1f}/100**")

        if score_1 > score_2:
            st.success("🎉 **첫 번째 사진이 더 동안입니다!**")
        elif score_1 < score_2:
            st.success("🎉 **두 번째 사진이 더 동안입니다!**")
        else:
            st.info("🤝 **두 사람의 동안 점수가 같습니다!**")
