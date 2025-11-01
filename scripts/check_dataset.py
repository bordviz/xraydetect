import yaml
from pathlib import Path
import cv2

class SIXRayDatasetChecker:
    def __init__(self, data_yaml_path):
        self.data_yaml_path = data_yaml_path
        self.data_config = None
        self.base_path = None


        
    def load_config(self):
        with open(self.data_yaml_path, 'r') as f:
            self.data_config = yaml.safe_load(f)
        
        self.base_path = Path(self.data_config['path'])
        print(f"📁 Базовый путь: {self.base_path}")
        print(f"🎯 Классы: {self.data_config['names']}")
        print(f"🔢 Количество классов: {self.data_config['nc']}")
    


    def check_images_and_labels(self, split='train'):
        print(f"\n🔍 Проверка {split}...")
        
        images_dir = self.base_path / split / 'images'
        labels_dir = self.base_path / split / 'labels'
        
        if not images_dir.exists():
            print(f"❌ Папка images не существует: {images_dir}")
            return
        if not labels_dir.exists():
            print(f"❌ Папка labels не существует: {labels_dir}")
            return
        
        image_files = list(images_dir.glob('*.*'))
        label_files = list(labels_dir.glob('*.txt'))
        
        print(f"📷 Изображений: {len(image_files)}")
        print(f"📝 Аннотаций: {len(label_files)}")
        
        image_stems = {f.stem for f in image_files}
        label_stems = {f.stem for f in label_files}
        
        missing_labels = image_stems - label_stems
        missing_images = label_stems - image_stems
        
        if missing_labels:
            print(f"⚠️ Отсутствуют аннотации для {len(missing_labels)} изображений")
        if missing_images:
            print(f"⚠️ Отсутствуют изображения для {len(missing_images)} аннотаций")
        
        for i, (img_file, lbl_file) in enumerate(zip(image_files[:3], label_files[:3])):
            print(f"\n📄 Пример {i+1}:")
            print(f"  Изображение: {img_file.name}")
            print(f"  Аннотация: {lbl_file.name}")
            
            img = cv2.imread(str(img_file))
            if img is not None:
                print(f"  Размер изображения: {img.shape}")
            else:
                print(f"  ❌ Ошибка загрузки изображения")
            
            with open(lbl_file, 'r') as f:
                lines = f.readlines()
                print(f"  Объектов в аннотации: {len(lines)}")
                for j, line in enumerate(lines[:2]):
                    cls, x, y, w, h = map(float, line.strip().split())
                    print(f"    Объект {j+1}: класс {int(cls)}, координаты [{x:.3f}, {y:.3f}, {w:.3f}, {h:.3f}]")
    


    def check_all_splits(self):
        self.load_config()
        
        for split in ['train', 'valid', 'test']:
            self.check_images_and_labels(split)

if __name__ == "__main__":
    checker = SIXRayDatasetChecker('data/SIXray/data.yaml')
    checker.check_all_splits()