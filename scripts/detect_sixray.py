import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from pathlib import Path

class SIXRayDetector:
    def __init__(self, model_path='runs/detect/train/weights/best.pt'):
        self.model = YOLO(model_path)
        self.class_names = ['gun', 'knife', 'wrench', 'pliers', 'scissors']
        self.colors = {
            'gun': (255, 0, 0),        # Красный
            'knife': (0, 255, 0),      # Зеленый
            'wrench': (0, 0, 255),     # Синий
            'pliers': (255, 255, 0),   # Голубой
            'scissors': (255, 0, 255)  # Пурпурный
        }
    
    def enhance_xray_image(self, image):
        """Улучшение рентгеновского изображения"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # CLAHE для улучшения контраста
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Конвертируем обратно в BGR
        if len(image.shape) == 3:
            enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        else:
            enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            
        return enhanced_bgr
    
    def detect(self, image_path, conf_threshold=0.25):
        """Детекция на одном изображении"""
        # Загрузка и улучшение изображения
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")
            
        enhanced_image = self.enhance_xray_image(image)
        
        # Детекция
        results = self.model.predict(
            enhanced_image,
            conf=conf_threshold,
            imgsz=640,
            augment=False
        )
        
        return results[0]
    
    def visualize_detection(self, image_path, output_path=None, conf_threshold=0.25):
        """Визуализация результатов детекции"""
        # Детекция
        result = self.detect(image_path, conf_threshold)
        
        # Визуализация
        plotted_image = result.plot()
        
        # Сохранение или отображение
        if output_path:
            cv2.imwrite(output_path, plotted_image)
            print(f"✅ Результат сохранен: {output_path}")
        
        # Отображение
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(plotted_image, cv2.COLOR_BGR2RGB))
        plt.title('Детекция запрещенных предметов - SIXray')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        # Статистика
        self.print_detection_stats(result)
        
        return result
    
    def print_detection_stats(self, result):
        """Вывод статистики детекции"""
        print("\n" + "="*60)
        print("📊 СТАТИСТИКА ДЕТЕКЦИИ SIXray")
        print("="*60)
        
        if len(result.boxes) == 0:
            print("❌ Объекты не обнаружены")
            return
            
        class_counts = {}
        for box in result.boxes:
            cls_id = int(box.cls.item())
            conf = box.conf.item()
            class_name = self.class_names[cls_id]
            
            if class_name not in class_counts:
                class_counts[class_name] = []
            class_counts[class_name].append(conf)
        
        print(f"📦 Всего обнаружено объектов: {len(result.boxes)}")
        print("\n📈 Детали по классам:")
        for class_name, confidences in class_counts.items():
            avg_conf = np.mean(confidences)
            max_conf = np.max(confidences)
            print(f"  🔸 {class_name}:")
            print(f"     Количество: {len(confidences)}")
            print(f"     Средняя уверенность: {avg_conf:.3f}")
            print(f"     Максимальная уверенность: {max_conf:.3f}")
        
        # Проверка безопасности
        dangerous_classes = [cls for cls in class_counts.keys()]
        if dangerous_classes:
            print(f"\n⚠️  ВНИМАНИЕ: Обнаружены запрещенные предметы!")
            for obj in dangerous_classes:
                print(f"   - {obj}")
        print("="*60)

def batch_process_sixray(input_dir, output_dir, model_path='best.pt'):
    """Пакетная обработка всех изображений в папке"""
    detector = SIXRayDetector(model_path)
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Поиск изображений
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(input_path.glob(ext))
        image_files.extend(input_path.glob(ext.upper()))
    
    print(f"🔄 Найдено {len(image_files)} изображений для обработки")
    
    for i, img_path in enumerate(image_files):
        print(f"\nОбработка {i+1}/{len(image_files)}: {img_path.name}")
        
        output_img_path = output_path / f"detected_{img_path.name}"
        
        try:
            detector.visualize_detection(
                str(img_path),
                output_path=str(output_img_path),
                conf_threshold=0.25
            )
        except Exception as e:
            print(f"❌ Ошибка при обработке {img_path}: {e}")

if __name__ == "__main__":
    # Пример использования
    detector = SIXRayDetector('runs/detect/train/weights/best.pt')
    
    # Детекция на одном изображении
    result = detector.visualize_detection(
        image_path='data/SIXray/test/images/example.jpg',
        output_path='result.jpg',
        conf_threshold=0.25
    )
    
    # Пакетная обработка
    # batch_process_sixray('data/SIXray/test/images', 'results')