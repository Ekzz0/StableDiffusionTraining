import logging
import os
from datetime import datetime


def setup_logger(model_name: str):
    # Создаем имя файла лога с датой и временем
    current_datetime = datetime.now().strftime("%d.%m.%Y_%H:%M:%S")
    log_file_name = f"train_{model_name.replace('/', '-').replace(' ', '_')}_{current_datetime}.log"
    log_path = os.path.join("logs", log_file_name)

    # Настраиваем логгер
    logger = logging.getLogger("notebook_logger")
    logger.setLevel(logging.INFO)

    # Файл для записи логов
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)

    # Формат логирования
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(formatter)

    # Добавляем обработчик в логгер
    logger.addHandler(file_handler)

    return logger
