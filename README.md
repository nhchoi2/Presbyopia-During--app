# 동안 vs 노안 판별기

AI를 활용하여 "내 얼굴이 동안일까, 노안일까?" 판별하고,  
친구와 비교하여 "누가 더 동안인가?" 점수를 매길 수 있는 웹 애플리케이션입니다.

---

## 주요 기능

- AI 기반 동안 vs 노안 판별
- 친구와 사진을 비교하여 동안 점수 대결
- 랜덤 피드백 제공 (유머 요소 추가)
- 사이드바에서 분석 모드 선택 가능 (개별 분석 / 친구 비교)
- 다크 모드 & 라이트 모드 지원

---

## 실행 방법

### 1. 필수 패키지 설치
먼저 `requirements.txt`를 사용하여 필요한 패키지를 설치합니다.

'pip install -r requirements.txt'

### 2. Streamlit 실행
아래 명령어를 실행하면 로컬 서버에서 앱을 실행할 수 있습니다.
'streamlit run app.py'

앱이 실행되면 브라우저에서 http://localhost:8501 주소로 접근할 수 있습니다.
```
📦 프로젝트 루트
├── 📂 model              # AI 모델 저장 폴더
│   ├── keras_model.h5    # 훈련된 Keras 모델
│   ├── labels.txt        # 모델의 분류 라벨
├── 📂 utils              # 기능 모듈 폴더
│   ├── face_detect.py    # 얼굴 검출 기능
│   ├── feedback.py       # 랜덤 피드백 메시지 제공
│   ├── sidebar.py        # 유지보수를 쉽게 하기 위한 사이드바 설정
│   ├── model_utils.py    # 모델 관련 함수 모음
├── 📂 images             # 앱 내에서 사용될 이미지 파일
│   ├── main.webp         # 메인화면 이미지
│   ├── sidebar.webp      # 사이드바 이미지
├── app.py                # 메인 애플리케이션 파일
├── requirements.txt      # 필요한 패키지 목록
├── README.md             # 프로젝트 설명 파일
```
UI 스크린샷
메인 화면

동안 vs 노안 결과

친구와 비교 기능

## 기술 스택
### 이 프로젝트는 다음과 같은 기술을 사용하여 개발되었습니다:

- Python 3.10+
- Streamlit - 웹 UI 개발
- TensorFlow/Keras - AI 모델 실행
- 향후 추가 기능 (업데이트 예정)
#### 이 앱은 지속적으로 개선될 예정이며, 다음 기능이 추가될 수 있습니다:
---
AI 기반 동안/노안 변환 필터 (동안 효과 or 노안 효과)
얼굴 감정 분석 (웃는 얼굴 vs 무표정 vs 화난 얼굴)
다양한 연령대별 동안 점수 평균 비교

개발자 정보
개발자: 최남호
이메일: choi1278@gmail.com
GitHub: [GitHub 방문하기](https://github.com/nhchoi2/Presbyopia-During--app)
