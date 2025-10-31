import os
from ultralytics import YOLO
import torch
import yaml

class SIXRayTrainer:
    def __init__(self, data_yaml_path, model_type='yolov8m.pt'):
        self.data_yaml_path = data_yaml_path
        self.model_type = model_type
        self.model = None
        
    def setup_environment(self):
        """Проверка окружения и данных"""
        print("🔍 Проверка окружения...")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            device = 0
        else:
            print("⚠️ Используется CPU")
            device = 'cpu'
            
        # Проверка существования data.yaml
        if not os.path.exists(self.data_yaml_path):
            raise FileNotFoundError(f"Файл {self.data_yaml_path} не найден!")
            
        # Загрузка конфигурации данных
        with open(self.data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
            
        # Проверка существования путей
        base_path = data_config['path']
        required_dirs = [
            os.path.join(base_path, 'train/images'),
            os.path.join(base_path, 'valid/images'), 
            os.path.join(base_path, 'test/images'),
            os.path.join(base_path, 'train/labels'),
            os.path.join(base_path, 'valid/labels'),
            os.path.join(base_path, 'test/labels')
        ]
        
        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                print(f"⚠️ Предупреждение: {dir_path} не существует")
                
        return device
    
    def setup_model(self):
        """Инициализация модели с учетом особенностей рентгеновских снимков"""
        print("🔄 Загрузка модели...")
        self.model = YOLO(self.model_type)
        
        # Настройка гиперпараметров для рентгеновских снимков
        self.model.overrides['imgsz'] = 640
        self.model.overrides['batch'] = 16
        self.model.overrides['epochs'] = 100
        self.model.overrides['lr0'] = 0.01
        self.model.overrides['weight_decay'] = 0.0005
        self.model.overrides['warmup_epochs'] = 3.0
        
        return self.model
    
    def train(self):
        """Основной метод обучения"""
        print("🎯 Начало обучения на датасете SIXray...")
        
        # Настройка окружения
        device = self.setup_environment()
        
        # Загрузка модели
        model = self.setup_model()
        
        # Обучение
        results = model.train(
            data=self.data_yaml_path,
            epochs=100,
            imgsz=640,
            batch=16,
            device=device,
            workers=8,
            optimizer='AdamW',
            lr0=0.001,
            patience=15,
            save=True,
            exist_ok=True,
            pretrained=True,
            mosaic=1.0,
            mixup=0.1,
            copy_paste=0.1,
            close_mosaic=10,
            overlap_mask=True,
            # Аугментации для рентгеновских снимков
            hsv_h=0.015,
            hsv_s=0.7, 
            hsv_v=0.4,
            translate=0.1,
            scale=0.5,
            fliplr=0.5,
            # Увеличенные веса для маленьких объектов
            box=7.5,
            cls=0.5,
            dfl=1.5
        )
        
        print("✅ Обучение завершено!")
        return results
    
    def validate(self):
        """Валидация модели"""
        if self.model is None:
            self.model = YOLO('runs/detect/train/weights/best.pt')
            
        metrics = self.model.val(
            data=self.data_yaml_path,
            split='test',
            imgsz=640,
            batch=16,
            conf=0.25,
            iou=0.45
        )
        
        print(f"📊 Результаты валидации:")
        print(f"mAP50: {metrics.box.map50:.4f}")
        print(f"mAP50-95: {metrics.box.map:.4f}") 
        print(f"Precision: {metrics.box.p:.4f}")
        print(f"Recall: {metrics.box.r:.4f}")
        
        return metrics

if __name__ == "__main__":
    # Укажите правильный путь к вашему data.yaml
    trainer = SIXRayTrainer(
        data_yaml_path='data/SIXray/data.yaml',
        model_type='yolov8m.pt'  # или 'yolov9c.pt' если используете YOLOv9
    )
    
    # Обучение
    results = trainer.train()
    
    # Валидация
    metrics = trainer.validate()