from dataclasses import dataclass
import torch

@dataclass
class InferenceSettings:
    guidance_scale: float  # Влияние текста на изображение
    num_inference_steps: int  # Большее количество шагов, больше качества
    height: int  # Размер изображения по высоте
    width: int  # Размер изображения по ширине
    generator: torch.Generator  # Генератор случайных чисел для воспроизводимости


@dataclass
class TrainingConfig:
    batch_size: int
    num_workers: int
    shuffle: bool
    max_tokenize_length: int
    image_resize: int
    model_name: str
    num_epochs: int
    log_interval: int
    weight_decay: float
    learning_rate: float
    lr_scheduler: str
    lr_warmup_steps: int
    device: str
    num_training_steps: int = None