import numpy as np
import tensorflow as tf
from PIL import Image

MODEL_PATH = "model/keras_model.h5"
LABELS_PATH = "model/labels.txt"

def load_model():
    """저장된 모델을 불러오는 함수"""
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

def load_labels():
    """라벨 파일을 불러오는 함수"""
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]

def preprocess_image(image):
    """이미지를 모델 입력 형태로 변환"""
    image_resized = image.resize((224, 224))
    image_array = np.asarray(image_resized).astype(np.float32) / 127.5 - 1
    return np.expand_dims(image_array, axis=0)

def predict_image(model, image, class_names):
    """이미지를 예측하는 함수"""
    data = preprocess_image(image)
    prediction = model.predict(data)
    index = np.argmax(prediction)
    confidence_score = prediction[0][index]
    return class_names[index], confidence_score
