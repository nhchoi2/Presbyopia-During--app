import cv2
import numpy as np
from PIL import Image
from deepface import DeepFace
from skimage.filters.rank import entropy
from skimage.morphology import disk

def detect_faces(image, use_dnn=True):
    """
    얼굴을 검출하는 함수 (기본적으로 DNN 모델 사용)
    """
    try:
        image_np = np.array(image)
        (h, w) = image_np.shape[:2]
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

        if use_dnn:
            # 🔹 OpenCV DNN 모델 사용 (더 높은 검출 정확도)
            net = cv2.dnn.readNetFromCaffe(
                cv2.data.haarcascades + "deploy.prototxt",
                cv2.data.haarcascades + "res10_300x300_ssd_iter_140000.caffemodel"
            )

            # DNN 전처리
            blob = cv2.dnn.blobFromImage(image_np, scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0))
            net.setInput(blob)
            detections = net.forward()

            faces = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:  # 🔹 신뢰도 50% 이상인 경우만 감지
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    faces.append((startX, startY, endX - startX, endY - startY))

        else:
            # 🔹 Haar Cascade 모델 사용 (기본 OpenCV 방식)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))

        return faces if len(faces) > 0 else []  # ✅ 항상 리스트 반환하여 오류 방지

    except Exception as e:
        print(f"❌ 얼굴 검출 중 오류 발생: {str(e)}")
        return []  # ✅ 오류 발생 시 빈 리스트 반환

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
    try:
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
        entropy_img = entropy(gray, disk(5))  # 피부 노화 분석
        wrinkle_score = entropy_img.mean()

        if wrinkle_score > 5.0:
            return "😔 주름이 많아 노안 경향이 있습니다."
        else:
            return "🎉 피부가 부드러워 동안으로 보입니다!"
    except Exception as e:
        return f"피부 분석 실패: {str(e)}"
