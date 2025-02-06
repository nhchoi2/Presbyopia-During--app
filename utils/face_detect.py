import cv2
import numpy as np
from PIL import Image

def crop_face(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    image_np = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        face = image_np[y:y+h, x:x+w]
        return Image.fromarray(face)  # 얼굴 부분만 반환
    return image  # 얼굴 인식 실패 시 원본 반환
