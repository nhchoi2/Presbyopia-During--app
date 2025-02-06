import cv2
import numpy as np
from PIL import Image
from deepface import DeepFace
from skimage.filters.rank import entropy
from skimage.morphology import disk

def detect_faces(image):
    """
    업로드된 이미지에서 얼굴을 검출하는 함수
    """
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # 이미지 변환 (PIL → OpenCV 배열)
    image_np = np.array(image)
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

    # 얼굴 검출 실행
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    return faces

def estimate_age(image):
    """
    DeepFace를 사용하여 얼굴의 나이를 예측하는 함수
    """
    try:
        analysis = DeepFace.analyze(image, actions=["age"], enforce_detection=False)
        estimated_age = analysis[0]["age"]
        return estimated_age
    except Exception as e:
        return f"나이 예측 실패: {str(e)}"

def analyze_skin(image):
    """
    주름(피부 상태)을 분석하는 함수
    """
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    entropy_img = entropy(gray, disk(5))  # 피부 노화 분석
    wrinkle_score = entropy_img.mean()

    if wrinkle_score > 5.0:
        return "😔 주름이 많아 노안 경향이 있습니다."
    else:
        return "🎉 피부가 부드러워 동안으로 보입니다!"
