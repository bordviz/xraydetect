import os
import shutil
from pathlib import Path

class MiniDatasetCreator:
    def __init__(self, source_dataset_path, target_dataset_path, num_images=100):
        """
        Args:
            source_dataset_path: путь к исходному датасету
            target_dataset_path: путь для нового мини-датасета  
            num_images: количество изображений для каждого сплита
        """
        self.source_path = Path(source_dataset_path)
        self.target_path = Path(target_dataset_path)
        self.num_images = num_images
        
    def create_mini_dataset(self):
        """Создает мини-датасет"""
        print(f"🚀 СОЗДАНИЕ МИНИ-ДАТАСЕТА ИЗ {self.num_images} ИЗОБРАЖЕНИЙ НА СПЛИТ")
        
        # Создаем структуру папок
        self._create_directory_structure()
        
        # Обрабатываем каждый сплит
        for split in ['train', 'valid', 'test']:
            print(f"\n📁 Обработка {split}...")
            self._process_split(split)
        
        # Создаем data.yaml для мини-датасета
        self._create_data_yaml()
        
        print(f"\n✅ МИНИ-ДАТАСЕТ СОЗДАН: {self.target_path}")
        
    def _create_directory_structure(self):
        """Создает структуру папок для мини-датасета"""
        directories = [
            'train/images', 'train/labels',
            'valid/images', 'valid/labels', 
            'test/images', 'test/labels'
        ]
        
        for dir_path in directories:
            full_path = self.target_path / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"📁 Создана папка: {full_path}")
    
    def _process_split(self, split):
        """Обрабатывает один сплит (train/valid/test)"""
        source_images_dir = self.source_path / split / 'images'
        source_labels_dir = self.source_path / split / 'labels'
        target_images_dir = self.target_path / split / 'images'
        target_labels_dir = self.target_path / split / 'labels'
        
        # Получаем список всех изображений
        image_files = list(source_images_dir.glob('*.*'))
        
        # Фильтруем только изображения
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = [f for f in image_files if f.suffix.lower() in valid_extensions]
        
        print(f"   Найдено изображений в исходном {split}: {len(image_files)}")
        
        # Берем первые N изображений
        selected_images = image_files[:self.num_images]
        print(f"   Выбрано изображений: {len(selected_images)}")
        
        # Копируем изображения и соответствующие аннотации
        copied_count = 0
        for img_path in selected_images:
            # Копируем изображение
            target_img_path = target_images_dir / img_path.name
            shutil.copy2(img_path, target_img_path)
            
            # Копируем аннотацию (если существует)
            label_path = source_labels_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                target_label_path = target_labels_dir / f"{img_path.stem}.txt"
                shutil.copy2(label_path, target_label_path)
                copied_count += 1
            else:
                print(f"   ⚠️ Аннотация не найдена: {label_path}")
        
        print(f"   📋 Скопировано пар изображение-аннотация: {copied_count}")
    
    def _create_data_yaml(self):
        """Создает data.yaml файл для мини-датасета"""
        # Читаем исходный data.yaml чтобы получить классы
        source_yaml = self.source_path / 'data.yaml'
        
        if source_yaml.exists():
            # Копируем исходный data.yaml
            target_yaml = self.target_path / 'data.yaml'
            shutil.copy2(source_yaml, target_yaml)
            print(f"📄 data.yaml скопирован: {target_yaml}")
        else:
            # Создаем простой data.yaml
            yaml_content = f"""# Mini SIXray dataset
path: {self.target_path}
train: train/images
val: valid/images
test: test/images

nc: 5
names:
  0: gun
  1: knife
  2: wrench
  3: pliers
  4: scissors

description: Mini SIXray dataset with {self.num_images} images per split
"""
            with open(self.target_path / 'data.yaml', 'w') as f:
                f.write(yaml_content)
            print(f"📄 data.yaml создан: {self.target_path / 'data.yaml'}")

def main():
    # ПУТИ - ЗАМЕНИТЕ НА СВОИ!
    source_dataset = "data/SIXray"  # Исходный датасет
    target_dataset = "data/SIXray_mini"  # Новый мини-датасет
    num_images = 100  # Количество изображений на каждый сплит
    
    print("🚀 СОЗДАНИЕ МИНИ-ДАТАСЕТА SIXray")
    print("=" * 50)
    print(f"📁 Источник: {source_dataset}")
    print(f"📁 Цель: {target_dataset}")
    print(f"🖼️ Изображений на сплит: {num_images}")
    print("=" * 50)
    
    # Проверяем существование исходного датасета
    if not Path(source_dataset).exists():
        print(f"❌ Исходный датасет не найден: {source_dataset}")
        print("🔧 Убедитесь, что путь правильный!")
        return
    
    creator = MiniDatasetCreator(os.path.abspath(source_dataset), os.path.abspath(target_dataset), num_images)
    creator.create_mini_dataset()
    
    # Выводим статистику
    print(f"\n📊 СТАТИСТИКА МИНИ-ДАТАСЕТА:")
    print(f"   Всего изображений: {num_images * 3}")
    print(f"   Train: {num_images} изображений")
    print(f"   Valid: {num_images} изображений") 
    print(f"   Test: {num_images} изображений")
    print(f"   Размер: ~{num_images * 3 * 0.5:.1f} MB (приблизительно)")

if __name__ == "__main__":
    main()