# Перенаправляем вывод tqdm в лог
import logging

import torch
from datasets import Dataset
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from transformers import CLIPTokenizer


class DataPreprocessor:
    def __init__(self, tokenizer: CLIPTokenizer, logger=None):
        self.max_length = None
        self.size = None
        self.tokenizer = tokenizer

        self.logger = logger or logging.getLogger(__name__)

    def _tokenize_data(self, element):
        """Функция токенизации текста. Создает токены по ключу input_ids"""
        text = element["caption"]
        # Токенизируем текст
        tokenized_text = self.tokenizer(
            text, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        element["input_ids"] = tokenized_text.input_ids.squeeze()  # Убираем лишнюю размерность

        return element

    def _preprocess_images(self, element, std: float = 0.5, mean: float = 0.5):
        """Функция для обработки изображений. Создает байты изображений по ключу pixel_values"""
        image = Image.open(element["image"]).convert("RGB")

        # Преобразования для изображений
        transform = Compose(
            [
                Resize([self.size, self.size]),  # Изменяем размер изображения на size x size
                ToTensor(),  # Преобразуем в тензор
                Normalize(
                    mean=mean.tolist(), std=std.tolist()
                ),  # TODO: подумать над нормализацией. Возможно из-за нее плывут цвета
            ]
        )
        element["pixel_values"] = transform(image)

        return element

    @staticmethod
    def _collate_fn(batch: dict):
        """Метод для распаковки батча"""
        pixel_values = torch.stack([torch.tensor(item["pixel_values"], dtype=torch.float32) for item in batch])
        input_ids = torch.stack([torch.tensor(item["input_ids"], dtype=torch.long) for item in batch])

        return {"pixel_values": pixel_values, "input_ids": input_ids}

    def _compute_mean_std(self, dataset):
        """Вычисление среднего и стандартного отклонения для изображений в dataset."""
        # Преобразования для изменения размера и конвертации в тензор

        transform = Compose(
            [
                Resize([self.size, self.size]),  # Изменяем размер изображения на self.size x self.size
                ToTensor(),  # Преобразуем изображение в тензор
            ]
        )

        all_tensors = []

        for element in dataset:
            # Преобразуем изображение в тензор с нужным размером
            image = Image.open(element["image"]).convert("RGB")
            tensor = transform(image)
            all_tensors.append(tensor)

        # Объединяем все тензоры в один
        all_tensors = torch.stack(all_tensors)

        # Вычисление среднего и стандартного отклонения по каналам (C, H, W)
        mean = all_tensors.mean([0, 2, 3])  # Среднее по всем изображениям по каналам
        std = all_tensors.std([0, 2, 3])  # Стандартное отклонение по всем изображениям по каналам

        return mean, std

    def transform(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        num_workers: int = 0,
        shuffle: bool = True,
        max_tokenize_length: int = 100,
        image_resize: int = 256,
    ) -> DataLoader:
        """Метод для применения всех преобразований к датасету"""

        self.max_length = max_tokenize_length
        self.size = image_resize

        # Пример: вычисление среднего и стандартного отклонения для вашего набора данных
        mean, std = self._compute_mean_std(dataset)
        self.logger.info("***** Starting tokenization and Image preprocessing... *****")
        dataset = dataset.map(self._tokenize_data, desc="tokenize_data")
        dataset = dataset.map(self._preprocess_images, desc="preprocess_images", fn_kwargs={"mean": mean, "std": std})
        self.logger.info("Tokenization and Image preprocessing completed.")

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=self._collate_fn,
            num_workers=num_workers,
            shuffle=shuffle,
        )

        return dataloader
