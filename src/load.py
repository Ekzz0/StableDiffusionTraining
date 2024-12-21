import os

import pandas as pd
from datasets import Dataset
from PIL import Image


def validate_images(row):
    """Метод проверяет корректность загруженных файлов"""
    try:
        img = Image.open(row["image"])
        img.verify()
        return True
    except Exception as e:
        print(f"Ошибка с ф`айлом {row['image']}: {e}")
        return False


def load_fantastic_dataset(filename: str, path_to_images: str) -> Dataset:
    """Метод загружает csv и изображения. Проверяет корректность изображений и возвращает Dataset"""
    df = pd.read_csv(filename)
    df["image"] = df["image"].apply(lambda x: os.path.join(path_to_images, x))

    # Проверяем что картинки загружены корректно
    df = df[df.apply(validate_images, axis=1)]

    # Преобразуем DataFrame в формат Hugging Face Dataset
    dataset = Dataset.from_pandas(df)

    return dataset

def load_drones_dataset(image_folder: str) -> Dataset:
    """Метод загружает изображения и возвращает Dataset"""
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg') or f.endswith('.png')]

    data = []

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        
        caption = os.path.splitext(image_file)[0]  
        
        data.append({
            "image": image_path,
            "caption": caption
        })

    dataset = Dataset.from_dict({
        "image": [item["image"] for item in data],
        "caption": [item["caption"] for item in data]
    })

    return dataset
