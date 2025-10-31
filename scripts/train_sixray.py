#!/usr/bin/env python3
"""
Скрипт обучения модели на датасете SIXray с принудительным использованием GPU
Путь: scripts/train_sixray.py
"""

import os
import sys
from pathlib import Path
import torch
import torch.cuda as cuda

# Добавляем корневую папку проекта в Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from ultralytics import YOLO
import yaml
import argparse

class SIXRayTrainer:
    def __init__(self, data_yaml_path, model_type='yolov8m.pt'):
        self.data_yaml_path = Path(data_yaml_path)
        self.model_type = model_type
        self.model = None
        self.project_root = Path(__file__).parent.parent
        
    def setup_environment(self):
        """Проверка окружения и принудительная настройка GPU"""
        print("🔍 ПРОВЕРКА ОКРУЖЕНИЯ И НАСТРОЙКА GPU...")
        
        # Принудительная проверка и настройка GPU
        print(f"🐍 PyTorch version: {torch.__version__}")
        print(f"⚡ CUDA available: {cuda.is_available()}")
        
        device = 'gpu'  # по умолчанию
        
        if cuda.is_available():
            # Получаем информацию о GPU
            gpu_count = cuda.device_count()
            print(f"🎮 Доступно GPU: {gpu_count}")
            
            for i in range(gpu_count):
                gpu_name = cuda.get_device_name(i)
                gpu_memory = cuda.get_device_properties(i).total_memory / 1e9
                print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            
            # Принудительно устанавливаем GPU
            device = 0  # используем первую GPU
            torch.cuda.set_device(device)
            
            # Проверяем, что GPU действительно используется
            current_device = cuda.current_device()
            print(f"✅ Используется GPU: {current_device} - {cuda.get_device_name(current_device)}")
            
            # Тест GPU
            test_tensor = torch.tensor([1.0, 2.0, 3.0]).cuda()
            print(f"✅ GPU тест пройден: тензор на устройстве {test_tensor.device}")
            
        else:
            print("❌ CUDA недоступна! Проверьте:")
            print("   - Установлен ли CUDA Toolkit")
            print("   - Установлен ли torch с поддержкой CUDA")
            print("   - Доступны ли драйверы NVIDIA")
            print("   - Запущен ли Docker с поддержкой GPU (если используется)")
            
        # Проверка существования data.yaml
        if not self.data_yaml_path.exists():
            raise FileNotFoundError(f"❌ Файл {self.data_yaml_path} не найден!")
            
        # Проверка структуры датасета
        with open(self.data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
            
        base_path = Path(data_config['path'])
        print(f"📁 Базовый путь датасета: {base_path}")
        
        # Проверка существования всех необходимых папок
        required_dirs = [
            base_path / 'train' / 'images',
            base_path / 'train' / 'labels',
            base_path / 'valid' / 'images', 
            base_path / 'valid' / 'labels',
            base_path / 'test' / 'images'
        ]
        
        for dir_path in required_dirs:
            if not dir_path.exists():
                print(f"⚠️  Предупреждение: {dir_path} не существует")
            else:
                file_count = len(list(dir_path.glob('*')))
                print(f"✅ {dir_path}: {file_count} файлов")
                
        return device
    
    def setup_model(self, device):
        """Инициализация модели с принудительной загрузкой на GPU"""
        print("\n🔄 ЗАГРУЗКА МОДЕЛИ НА GPU...")
        
        # Путь к предобученной модели
        model_path = self.project_root / 'models' / self.model_type
        
        if model_path.exists():
            print(f"✅ Используется локальная модель: {model_path}")
            
            # Загружаем модель с явным указанием устройства
            if device != 'cpu':
                self.model = YOLO(str(model_path)).to('cuda')
            else:
                self.model = YOLO(str(model_path))
        else:
            print(f"📥 Загрузка модели {self.model_type}...")
            
            # Загружаем модель с явным указанием устройства
            if device != 'cpu':
                self.model = YOLO(self.model_type).to('cuda')
            else:
                self.model = YOLO(self.model_type)
        
        # Проверяем, где находится модель
        print(f"📍 Модель загружена на: {next(self.model.model.parameters()).device}")
            
        return self.model
    
    def force_gpu_usage(self):
        """Принудительная настройка для использования GPU"""
        if cuda.is_available():
            # Очищаем кэш CUDA
            cuda.empty_cache()
            
            # Устанавливаем флаги для максимальной производительности
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            
            print("✅ CUDA оптимизации включены")
        else:
            print("⚠️  CUDA недоступна, используются CPU вычисления")
    
    def train(self, epochs=100, imgsz=640, batch=16):
        """Основной метод обучения с принудительным использованием GPU"""
        print("\n🎯 ЗАПУСК ОБУЧЕНИЯ НА SIXray С ИСПОЛЬЗОВАНИЕМ GPU...")
        
        # Принудительная настройка GPU
        self.force_gpu_usage()
        
        # Настройка окружения
        device = self.setup_environment()
        
        # Загрузка модели на правильное устройство
        model = self.setup_model(device)
        
        # Пути для сохранения результатов
        runs_dir = self.project_root / 'runs'
        runs_dir.mkdir(exist_ok=True)
        
        print(f"💾 Результаты будут сохранены в: {runs_dir}")
        
        # Параметры для обучения с GPU
        train_kwargs = {
            'data': str(self.data_yaml_path),
            'epochs': epochs,
            'imgsz': imgsz,
            'batch': batch,
            'workers': 8,
            'optimizer': 'AdamW',
            'lr0': 0.001,
            'patience': 15,
            'save': True,
            'exist_ok': True,
            'pretrained': True,
            'project': str(runs_dir),
            'name': 'detect/train_sixray_gpu',
            # Аугментации для рентгеновских снимков
            'hsv_h': 0.015,
            'hsv_s': 0.7, 
            'hsv_v': 0.4,
            'translate': 0.1,
            'scale': 0.5,
            'fliplr': 0.5,
            # Регуляризация
            'weight_decay': 0.0005,
            'warmup_epochs': 3.0,
            # Приоритеты для детекции
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5
        }
        
        # Явно указываем устройство для обучения
        if device != 'cpu':
            train_kwargs['device'] = device
            print(f"🚀 Обучение на GPU: {device}")
        else:
            train_kwargs['device'] = 'cpu'
            print("⚠️  Обучение на CPU (GPU недоступна)")
        
        # Обучение модели
        print("🔥 НАЧАЛО ОБУЧЕНИЯ...")
        results = model.train(**train_kwargs)
        
        print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
        
        # Путь к лучшей модели
        best_model_path = runs_dir / 'detect' / 'train_sixray_gpu' / 'weights' / 'best.pt'
        print(f"🏆 Лучшая модель сохранена: {best_model_path}")
        
        return results, best_model_path

def check_cuda_installation():
    """Проверка установки CUDA и совместимости"""
    print("🔍 ДЕТАЛЬНАЯ ПРОВЕРКА CUDA...")
    
    # Проверка версий
    print(f"PyTorch CUDA доступна: {cuda.is_available()}")
    if cuda.is_available():
        print(f"PyTorch CUDA версия: {torch.version.cuda}")
        print(f"Количество GPU: {cuda.device_count()}")
        
        # Проверка каждой GPU
        for i in range(cuda.device_count()):
            print(f"GPU {i}: {cuda.get_device_name(i)}")
            print(f"  Память: {cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
            print(f"  Вычислительная способность: {cuda.get_device_properties(i).major}.{cuda.get_device_properties(i).minor}")
    
    # Проверка через nvidia-smi (если доступно)
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ nvidia-smi доступен")
            # Выводим первую строку для информации
            lines = result.stdout.split('\n')
            if len(lines) > 0:
                print(f"  nvidia-smi: {lines[0]}")
        else:
            print("❌ nvidia-smi недоступен")
    except:
        print("❌ nvidia-smi не установлен")

def main():
    parser = argparse.ArgumentParser(description='Обучение YOLO на SIXray с GPU')
    parser.add_argument('--epochs', type=int, default=100, help='Количество эпох')
    parser.add_argument('--batch', type=int, default=16, help='Размер батча')
    parser.add_argument('--imgsz', type=int, default=640, help='Размер изображения')
    parser.add_argument('--model', type=str, default='yolov8m.pt', help='Модель YOLO')
    parser.add_argument('--check-cuda', action='store_true', help='Проверить установку CUDA')
    
    args = parser.parse_args()
    
    if args.check_cuda:
        check_cuda_installation()
        return
    
    # АБСОЛЮТНЫЙ путь к data.yaml
    data_yaml_path = "/Users/bordvizov/Desktop/Python/xraydetect/data/SIXray/data.yaml"
    
    print("🚀 ЗАПУСК СКРИПТА ОБУЧЕНИЯ С GPU")
    print("=" * 50)
    
    trainer = SIXRayTrainer(
        data_yaml_path=data_yaml_path,
        model_type=args.model
    )
    
    try:
        results, best_model = trainer.train(
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz
        )
        print(f"\n🎉 Обучение успешно завершено!")
        print(f"📁 Лучшая модель: {best_model}")
        
    except Exception as e:
        print(f"❌ Ошибка при обучении: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()