import sys
from pathlib import Path
import torch
import torch.cuda as cuda
import os

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from ultralytics import YOLO
import yaml
import argparse

data_yaml_path = "data/SIXray_mini/data.yaml"
absolute_path = os.path.abspath(data_yaml_path)

class SIXRayTrainer:
    def __init__(self, data_yaml_path, model_type='yolov8m.pt'):
        self.data_yaml_path = Path(data_yaml_path)
        self.model_type = model_type
        self.model = None
        self.project_root = Path(__file__).parent.parent
        
    def setup_environment(self):
        print("🔍 ПРОВЕРКА ОКРУЖЕНИЯ И НАСТРОЙКА УСТРОЙСТВА...")
        
        print(f"🐍 PyTorch version: {torch.__version__}")
        print(f"⚡ CUDA available: {cuda.is_available()}")
        print(f"🍎 MPS available: {torch.backends.mps.is_available()}")
        
        if cuda.is_available():
            device = 0
            gpu_count = cuda.device_count()
            print(f"🎮 Доступно GPU: {gpu_count}")
            
            for i in range(gpu_count):
                gpu_name = cuda.get_device_name(i)
                gpu_memory = cuda.get_device_properties(i).total_memory / 1e9
                print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            
            torch.cuda.set_device(device)
            
            current_device = cuda.current_device()
            print(f"✅ Используется CUDA GPU: {current_device} - {cuda.get_device_name(current_device)}")
            
            test_tensor = torch.tensor([1.0, 2.0, 3.0]).cuda()
            print(f"✅ CUDA тест пройден: тензор на устройстве {test_tensor.device}")
            
        elif torch.backends.mps.is_available():
            device = 'mps'
            print("✅ Используется MPS (Metal Performance Shaders) на macOS")
            
            test_tensor = torch.tensor([1.0, 2.0, 3.0]).to('mps')
            print(f"✅ MPS тест пройден: тензор на устройстве {test_tensor.device}")
            
        else:
            device = 'cpu'
            print("❌ CUDA и MPS недоступны! Используется CPU.")
            print("   Для ускорения обучения рекомендуется:")
            print("   - Установить torch с поддержкой CUDA (для NVIDIA GPU)")
            print("   - Использовать Mac с Apple Silicon (для MPS)")
            print("   - Проверить установку драйверов")
            
        if not self.data_yaml_path.exists():
            raise FileNotFoundError(f"❌ Файл {self.data_yaml_path} не найден!")
            
        with open(self.data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
            
        base_path = Path(data_config['path'])
        print(f"📁 Базовый путь датасета: {base_path}")
        
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
        print("\n🔄 ЗАГРУЗКА МОДЕЛИ НА УСТРОЙСТВО...")
        
        model_path = self.project_root / 'models' / self.model_type
        
        if model_path.exists():
            print(f"✅ Используется локальная модель: {model_path}")
            
            if device == 'mps':
                self.model = YOLO(str(model_path)).to('mps')
            elif device != 'cpu':
                self.model = YOLO(str(model_path)).to('cuda')
            else:
                self.model = YOLO(str(model_path))
        else:
            print(f"📥 Загрузка модели {self.model_type}...")
            
            if device == 'mps':
                self.model = YOLO(self.model_type).to('mps')
            elif device != 'cpu':
                self.model = YOLO(self.model_type).to('cuda')
            else:
                self.model = YOLO(self.model_type)
        
        model_device = next(self.model.model.parameters()).device
        print(f"📍 Модель загружена на: {model_device}")
            
        return self.model
    
    def optimize_performance(self, device):
        print("⚡ НАСТРОЙКА ОПТИМИЗАЦИЙ ДЛЯ УСТРОЙСТВА...")
        
        if device != 'cpu':
            if device == 'mps':
                torch.mps.empty_cache()
                print("✅ Кэш MPS очищен")
            else:
                cuda.empty_cache()
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.enabled = True
                print("✅ CUDA оптимизации включены")
        else:
            print("⚠️  CPU режим, оптимизации GPU не применяются")
    
    def train(self, epochs=100, imgsz=640, batch=16):
        print("\n🎯 ЗАПУСК ОБУЧЕНИЯ НА SIXray...")
        
        device = self.setup_environment()
        
        self.optimize_performance(device)
        
        model = self.setup_model(device)
        
        runs_dir = self.project_root / 'runs'
        runs_dir.mkdir(exist_ok=True)
        
        print(f"💾 Результаты будут сохранены в: {runs_dir}")
        
        train_kwargs = {
            'data': str(self.data_yaml_path),
            'epochs': epochs,
            'imgsz': imgsz,
            'batch': batch,
            'save': True,
            'exist_ok': True,
            'pretrained': True,
            'project': str(runs_dir),
            'name': 'detect/train_sixray',
            'hsv_h': 0.015,
            'hsv_s': 0.7, 
            'hsv_v': 0.4,
            'translate': 0.1,
            'scale': 0.5,
            'fliplr': 0.5,
            'weight_decay': 0.0005,
            'warmup_epochs': 3.0,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5
        }
        
        if device == 'mps':
            train_kwargs.update({
                'device': 'mps',
                'workers': 8, 
                'optimizer': 'AdamW',
            })
            print("🚀 Обучение на MPS (macOS GPU)")
            
        elif device != 'cpu':
            train_kwargs.update({
                'device': device,
                'workers': 8,
                'optimizer': 'AdamW',
                'lr0': 0.001,
                'patience': 15,
            })
            print(f"🚀 Обучение на CUDA GPU: {device}")
            
        else:
            train_kwargs.update({
                'device': 'cpu',
                'workers': 8, 
                'optimizer': 'Adam',
            })
            print("⚠️  Обучение на CPU (GPU недоступна)")
        
        print("🔥 НАЧАЛО ОБУЧЕНИЯ...")
        try:
            results = model.train(**train_kwargs)
            print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
            
            best_model_path = runs_dir / 'detect' / 'train_sixray' / 'weights' / 'best.pt'
            print(f"🏆 Лучшая модель сохранена: {best_model_path}")
            
            return results, best_model_path
            
        except RuntimeError as e:
            if "MPS" in str(e) and device == 'mps':
                print("❌ Ошибка MPS. Пытаемся перезапустить на CPU...")
                train_kwargs['device'] = 'cpu'
                train_kwargs['workers'] = 4
                print("🔄 Перезапуск обучения на CPU...")
                results = model.train(**train_kwargs)
                
                best_model_path = runs_dir / 'detect' / 'train_sixray' / 'weights' / 'best.pt'
                return results, best_model_path
            else:
                raise e
    
    def get_device_info(self):
        device_info = {
            'cuda_available': cuda.is_available(),
            'mps_available': torch.backends.mps.is_available(),
            'devices': []
        }
        
        if cuda.is_available():
            for i in range(cuda.device_count()):
                device_info['devices'].append({
                    'type': 'cuda',
                    'index': i,
                    'name': cuda.get_device_name(i),
                    'memory_gb': cuda.get_device_properties(i).total_memory / 1e9
                })
        
        if torch.backends.mps.is_available():
            device_info['devices'].append({
                'type': 'mps',
                'index': 'mps',
                'name': 'Apple Metal Performance Shaders',
                'memory_gb': 'unknown'
            })
            
        return device_info

def check_device_installation():
    print("🔍 ДЕТАЛЬНАЯ ПРОВЕРКА УСТРОЙСТВ...")
    
    trainer = SIXRayTrainer(absolute_path)
    device_info = trainer.get_device_info()
    
    print(f"PyTorch CUDA доступна: {device_info['cuda_available']}")
    print(f"PyTorch MPS доступен: {device_info['mps_available']}")
    
    if device_info['cuda_available']:
        print(f"Количество GPU: {len([d for d in device_info['devices'] if d['type'] == 'cuda'])}")
        
        # Проверка каждой GPU
        for device in device_info['devices']:
            if device['type'] == 'cuda':
                print(f"GPU {device['index']}: {device['name']}")
                print(f"  Память: {device['memory_gb']:.1f} GB")
    
    if device_info['mps_available']:
        print("✅ MPS (macOS GPU) доступен для использования")
        
    if not device_info['cuda_available'] and not device_info['mps_available']:
        print("❌ Ни CUDA ни MPS не доступны! Доступен только CPU.")

def main():
    parser = argparse.ArgumentParser(description='Обучение YOLO на SIXray с поддержкой GPU/CUDA/MPS')
    parser.add_argument('--epochs', type=int, default=100, help='Количество эпох')
    parser.add_argument('--batch', type=int, default=16, help='Размер батча')
    parser.add_argument('--imgsz', type=int, default=640, help='Размер изображения')
    parser.add_argument('--model', type=str, default='yolov8m.pt', help='Модель YOLO')
    parser.add_argument('--check-devices', action='store_true', help='Проверить доступные устройства')
    parser.add_argument('--force-cpu', action='store_true', help='Принудительно использовать CPU')
    
    args = parser.parse_args()
    
    if args.check_devices:
        check_device_installation()
        return
    
    # АБСОЛЮТНЫЙ путь к data.yaml
    # data_yaml_path = "/Users/bordvizov/Desktop/Python/xraydetect/data/SIXray/data.yaml"
    # data_yaml_path = "../../data/SIXray/data.yaml"
    
    print("🚀 ЗАПУСК СКРИПТА ОБУЧЕНИЯ С ПОДДЕРЖКОЙ GPU/CUDA/MPS")
    print("=" * 60)
    
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