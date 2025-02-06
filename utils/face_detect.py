import cv2
import numpy as np
from PIL import Image

def detect_and_crop_face(image: Image.Image) -> Image.Image:
    """
    업로드된 PIL 이미지에서 얼굴을 자동으로 감지하고,
    가장 큰 얼굴 영역을 잘라서 반환합니다.
    얼굴이 감지되지 않으면 원본 이미지를 반환합니다.
    """
    # PIL 이미지를 OpenCV용 numpy 배열로 변환
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # OpenCV에서 제공하는 기본 분류기 (haar cascade 등) 사용 예시
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # 얼굴 탐지
    faces = face_cascade.detectMultiScale(cv_image, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        return image  # 얼굴이 없으면 원본 그대로 반환

    # 가장 큰 얼굴 영역만 선택
    x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])

    # 얼굴 영역으로 크롭
    cropped_cv_image = cv_image[y:y+h, x:x+w]
    cropped_pil_image = Image.fromarray(cv2.cvtColor(cropped_cv_image, cv2.COLOR_BGR2RGB))

    return cropped_pil_image
