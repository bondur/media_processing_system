import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
from PIL import Image, ImageEnhance

class ImageProcessor:
    def __init__(self, models_path="models/"):
        self.models_path = models_path
        self.enhancement_model = self._load_enhancement_model()
        self.classification_model = self._load_classification_model()
    
    def _load_enhancement_model(self):
        model_path = os.path.join(self.models_path, "enhancement_model.h5")
        if os.path.exists(model_path):
            try:
                return load_model(model_path)
            except:
                print("⚠️ Не удалось загрузить модель улучшения")
        return None
    
    def _load_classification_model(self):
        model_path = os.path.join(self.models_path, "classification_model.h5")
        if os.path.exists(model_path):
            try:
                return load_model(model_path)
            except:
                print("⚠️ Не удалось загрузить модель классификации")
        return None
    
    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """Улучшение качества изображения"""
        if self.enhancement_model:
            # Предобработка для нейронной сети
            processed = cv2.resize(image, (256, 256))
            processed = processed.astype('float32') / 255.0
            processed = np.expand_dims(processed, axis=0)
            
            # Обработка моделью
            enhanced = self.enhancement_model.predict(processed)[0]
            enhanced = (enhanced * 255).astype('uint8')
            return cv2.resize(enhanced, (image.shape[1], image.shape[0]))
        else:
            # Fallback: традиционные методы улучшения
            return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    
    def intelligent_compression(self, image: np.ndarray) -> tuple:
        """Адаптивное сжатие изображения"""
        # Анализ сложности изображения
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gradient = cv2.Sobel(gray, cv2.CV_64F, 1, 1)
        avg_gradient = np.mean(np.abs(gradient))
        
        # Расчет качества сжатия (70-95)
        quality = int(np.clip(avg_gradient/10 + 70, 75, 95))
        
        # Применение сжатия
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, buffer = cv2.imencode('.jpg', image, encode_param)
        compressed = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        
        return compressed, {"quality": quality, "size_kb": len(buffer)//1024}
    
    def classify_image(self, image: np.ndarray) -> list:
        """Классификация изображения"""
        if self.classification_model:
            # Предобработка для классификации
            processed = cv2.resize(image, (224, 224))
            processed = processed.astype('float32') / 255.0
            processed = np.expand_dims(processed, axis=0)
            
            # Классификация
            predictions = self.classification_model.predict(processed)
            return self._decode_predictions(predictions)
        return ["Модель классификации недоступна"]
    
    def _decode_predictions(self, predictions, top=3):
        """Декодирование результатов классификации"""
        # Заглушка: в реальном проекте здесь будет логика декодирования
        classes = ["Пейзаж", "Портрет", "Животные", "Транспорт", "Еда"]
        return classes[:top]