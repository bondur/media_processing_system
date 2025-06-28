import cv2
import numpy as np

class AutoCropper:
    def __init__(self):
        try:
            # Для новых версий OpenCV (4.5+)
            saliency = cv2.saliency.Saliency_create()
            if hasattr(saliency, 'StaticSaliencySpectralResidual_create'):
                self.saliency_detector = saliency.StaticSaliencySpectralResidual_create()
            else:
                self.saliency_detector = cv2.saliency.StaticSaliencySpectralResidual_create()
        except Exception as e:
            # Резервный вариант
            self.saliency_detector = None
            print(f"Saliency detector not available: {e}, using alternative methods")
    
    def find_optimal_crop(self, image: np.ndarray) -> tuple:
        """Поиск оптимального кадрирования"""
        # Если детектор не доступен, используем кадрирование по умолчанию
        if self.saliency_detector is None:
            return self._default_crop(image)
        
        # Создание карты значимости
        success, saliency_map = self.saliency_detector.computeSaliency(image)
        if not success:
            return self._default_crop(image)
        
        # Бинаризация карты значимости
        _, saliency_thresh = cv2.threshold(
            (saliency_map * 255).astype("uint8"), 
            0, 255, 
            cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )
        
        # Поиск контуров
        contours, _ = cv2.findContours(
            saliency_thresh, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return self._default_crop(image)
        
        # Нахождение основного объекта
        main_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(main_contour)
        
        # Добавление отступов
        padding = 20
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)
        
        return x, y, w, h
    
    def _default_crop(self, image: np.ndarray) -> tuple:
        """Кадрирование по умолчанию (центр)"""
        h, w = image.shape[:2]
        crop_w = w * 0.8
        crop_h = h * 0.8
        x = (w - crop_w) // 2
        y = (h - crop_h) // 2
        return int(x), int(y), int(crop_w), int(crop_h)