import streamlit as st

def load_sidebar():
    """
    Streamlit 사이드바 UI를 로드하는 함수
    """
    st.sidebar.title("🔧 설정 메뉴")
    st.sidebar.info("분석 모드를 선택하고, 얼굴 비교 기능을 활용해 보세요!")

    # 🔹 분석 모드 선택
    mode = st.sidebar.radio("🔍 분석 모드 선택", ["개별 분석", "친구와 비교"])

    # 🔹 도움말 제공
    st.sidebar.markdown("📌 **사용 방법**")
    st.sidebar.markdown("1️⃣ 개별 분석: 사진을 업로드하면 AI가 동안/노안을 판별합니다.")
    st.sidebar.markdown("2️⃣ 친구와 비교: 두 개의 사진을 업로드하여 동안 점수를 비교합니다.")
    st.sidebar.markdown("---")
    
    # 🔹 테마 선택 (추후 추가 가능)
    st.sidebar.markdown("🎨 **앱 테마 설정**")
    theme = st.sidebar.selectbox("테마 선택", ["기본", "다크 모드", "라이트 모드"])

    return mode, theme
