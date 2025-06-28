from flask import Flask, request, render_template, jsonify
import cv2
import numpy as np
import base64
import os
from image_processor import ImageProcessor
from auto_cropper import AutoCropper

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# Инициализация обработчиков
processor = ImageProcessor(models_path="models/")
cropper = AutoCropper()

@app.route('/')
def index():
    """Главная страница с веб-интерфейсом"""
    return render_template('index.html')

@app.route('/api/process', methods=['POST'])
def process_image():
    """API для обработки изображений"""
    try:
        # Проверка наличия файла
        if 'image' not in request.files:
            return jsonify({"error": "Изображение не предоставлено"}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "Файл не выбран"}), 400
        
        # Чтение изображения
        img_bytes = file.read()
        image = cv2.imdecode(
            np.frombuffer(img_bytes, np.uint8), 
            cv2.IMREAD_COLOR
        )
        if image is None:
            return jsonify({"error": "Неверный формат изображения"}), 400
        
        # Параметры обработки
        operations = {
            "enhance": request.form.get('enhance') == 'true',
            "compress": request.form.get('compress') == 'true',
            "crop": request.form.get('crop') == 'true',
            "classify": request.form.get('classify') == 'true'
        }
        
        # Обработка изображения
        processed_image = image.copy()
        result_info = {"original_size": f"{image.shape[1]}x{image.shape[0]}"}
        
        # Применение операций
        if operations["enhance"]:
            processed_image = processor.enhance_image(processed_image)
            result_info["enhancement"] = "Применено"
        
        if operations["compress"]:
            processed_image, comp_info = processor.intelligent_compression(processed_image)
            result_info["compression"] = comp_info
        
        if operations["crop"]:
            x, y, w, h = cropper.find_optimal_crop(processed_image)
            processed_image = processed_image[y:y+h, x:x+w]
            result_info["cropping"] = {
                "x": x, "y": y, "width": w, "height": h
            }
        
        if operations["classify"]:
            tags = processor.classify_image(processed_image)
            result_info["classification"] = tags
        
        # Кодирование результата
        _, buffer = cv2.imencode('.jpg', processed_image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        result_info["processed_image"] = f"data:image/jpeg;base64,{img_base64}"
        result_info["success"] = True
        
        return jsonify(result_info)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    os.makedirs("models", exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)