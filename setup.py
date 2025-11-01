import yaml
import os

def update_path_field(file_path, new_path_value):
    try:
        # Чтение файла
        with open(file_path, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file) or {}
        
        # Изменяем поле path
        data['path'] = new_path_value
        
        # Запись обратно в файл
        with open(file_path, 'w', encoding='utf-8') as file:
            yaml.dump(data, file, default_flow_style=False, allow_unicode=True, indent=2)
        
        print(f"Поле 'path' успешно обновлено на: {new_path_value}")
        
    except Exception as e:
        print(f"Ошибка при обновлении файла: {e}")

# Пример использования
if __name__ == "__main__":

    data_yaml_path = 'data/SIXray/data.yaml'
    data_yaml_mini_path = 'data/SIXray_mini/data.yaml'
    update_path_field(data_yaml_path, os.path.abspath('data/SIXray'))
    update_path_field(data_yaml_mini_path, os.path.abspath('data/SIXray_mini'))