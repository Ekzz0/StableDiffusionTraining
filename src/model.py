import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    EulerDiscreteScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from transformers import CLIPTextModel


class WarmupReduceLROnPlateau:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        reduce_scheduler: ReduceLROnPlateau,
        base_lr: float = 0.0,
    ):
        """
        Custom scheduler with warmup and ReduceLROnPlateau.
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.reduce_scheduler = reduce_scheduler
        self.current_step = 0
        self.base_lr = base_lr
        self.max_lr = [group["lr"] for group in optimizer.param_groups]

    def step(self, metrics=None):
        if self.current_step < self.warmup_steps:
            # Фаза warmup
            lr = self.base_lr + (self.max_lr[0] - self.base_lr) * (self.current_step / self.warmup_steps)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
        else:
            # Фаза ReduceLROnPlateau
            self.reduce_scheduler.step(metrics)

        self.current_step += 1

    def get_last_lr(self):
        """Возвращает текущий lr."""
        return [group["lr"] for group in self.optimizer.param_groups]


class CustomStableDiffusion:
    def __init__(self, logger):
        self.logger = logger

        # Заделка на будущее
        self.noise_scheduler = None
        self.text_encoder = None
        self.vae = None
        self.unet = None

        self.optimizer = None
        self.accelerator = None
        self.lr_scheduler = None
        self.weight_dtype = None
        self.epoch_losses = None
        self.reduce_scheduler = None

        self.pipeline = None

    def init_model(self, settings):
        """Инициализируем части stable diffusion"""

        self.noise_scheduler = DDPMScheduler.from_pretrained(settings.model_name, subfolder="scheduler", revision=None)
        self.text_encoder = CLIPTextModel.from_pretrained(settings.model_name, subfolder="text_encoder", revision=None)
        self.vae = AutoencoderKL.from_pretrained(settings.model_name, subfolder="vae", revision=None)
        self.unet = UNet2DConditionModel.from_pretrained(settings.model_name, subfolder="unet", revision=None)

        # Freeze vae and text_encoder
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        self.logger.info(f"Model {settings.model_name} are loaded!")

    def _prepare_to_train(self, train_dataloader, val_dataloader, settings):
        """Инициализирует все объекты для обучение unet"""

        self.optimizer = AdamW(
            self.unet.parameters(),
            lr=settings.learning_rate,
            weight_decay=settings.weight_decay,
        )
        self.accelerator = Accelerator()
        self.lr_scheduler = get_scheduler(
            settings.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=settings.lr_warmup_steps,
            num_training_steps=settings.num_training_steps,
        )
        # self.reduce_scheduler = ReduceLROnPlateau(
        #     optimizer=self.optimizer,
        #     mode="min",  
        #     factor=0.5,  
        #     patience=3,  
        #     threshold=0.01, 
        #     cooldown=0,  
        #     min_lr=1e-6,  
        # )
        # self.lr_scheduler = WarmupReduceLROnPlateau(
        #     self.optimizer, warmup_steps=settings.lr_warmup_steps, reduce_scheduler=self.reduce_scheduler, base_lr=1e-6
        # )

        self.unet, self.optimizer, train_dataloader, val_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.unet, self.optimizer, train_dataloader, val_dataloader, self.lr_scheduler
        )

        self.weight_dtype = torch.float32

        self.unet.to(self.accelerator.device, dtype=self.weight_dtype)
        self.text_encoder.to(self.accelerator.device, dtype=self.weight_dtype)
        self.vae.to(self.accelerator.device, dtype=self.weight_dtype)

        return train_dataloader, val_dataloader

    @staticmethod
    def _compute_loss(output, target, reduction="mean"):
        """MSE Loss"""
        return F.mse_loss(output, target, reduction=reduction)

    def _batch_train():
        pass

    def train_step(self, batch, settings):
        """преобразование изображений в латенты, добавление шума."""
        pixel_values = batch["pixel_values"].to(settings.device)
        text_input = batch["input_ids"].to(settings.device)

        # Преобразуем изображения в латентное пространство
        latents = self.vae.encode(pixel_values.to(self.weight_dtype)).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        # Генерируем шум
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]

        # Случайные временные шаги
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()

        # Добавляем шум в латенты
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Текстовые эмбеддинги для кондиционирования
        encoder_hidden_states = self.text_encoder(text_input)[0]

        return noisy_latents, timesteps, encoder_hidden_states, latents, noise

    def compute_target_step(self, latents, noise, timesteps):
        """Вычисление целевой переменной для потерь."""
        if self.noise_scheduler.config.prediction_type == "epsilon":
            return noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            return self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

    def compute_loss_step(self, noisy_latents, timesteps, encoder_hidden_states, target):
        """Вычисление потерь для одного шага."""
        output = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        loss = self._compute_loss(output.float(), target.float(), reduction="mean")
        return loss

    def fit(self, train_dataloader, val_dataloader, settings):
        """Обучение"""
        train_dataloader, val_dataloader = self._prepare_to_train(train_dataloader, val_dataloader, settings)

        self.logger.info(f"***** Start {settings.model_name} training... *****")

        self.epoch_losses = []
        self.val_losses = []

        for epoch in range(settings.num_epochs):
            self.unet.train()
            loop = tqdm(train_dataloader, total=len(train_dataloader), desc=f"Epoch {epoch + 1}")
            epoch_loss = 0

            for step, batch in enumerate(loop):
                with self.accelerator.accumulate(self.unet):
                    noisy_latents, timesteps, encoder_hidden_states, latents, noise = self.train_step(batch, settings)
                    target = self.compute_target_step(latents, noise, timesteps)
                    loss = self.compute_loss_step(noisy_latents, timesteps, encoder_hidden_states, target)

                    # Backpropagation
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.unet.parameters(), 1)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                    epoch_loss += loss.detach().item()

                    if step % settings.log_interval == 0:
                        self.logger.info(
                            f"Epoch {epoch}, Step {step}, Loss: {loss.detach().item()}, lr: {self.optimizer.param_groups[0]['lr']}"
                        )

            avg_epoch_loss = epoch_loss / len(train_dataloader)
            self.epoch_losses.append(avg_epoch_loss)
            val_loss = self.validate(val_dataloader, settings)
            self.val_losses.append(val_loss)

            self.logger.info(f"Epoch {epoch}, Epoch Loss: {avg_epoch_loss}, Validation Loss: {val_loss}")

            torch.cuda.empty_cache()

    def validate(self, dataloader, settings):
        """Валидация"""
        self.unet.eval()
        val_loss = 0.0

        with torch.no_grad():
            loop = tqdm(dataloader, total=len(dataloader), desc="Validation")
            for step, batch in enumerate(loop):
                noisy_latents, timesteps, encoder_hidden_states, latents, noise = self.train_step(batch, settings)
                target = self.compute_target_step(latents, noise, timesteps)
                loss = self.compute_loss_step(noisy_latents, timesteps, encoder_hidden_states, target)
                val_loss += loss.detach().item()

        avg_val_loss = val_loss / len(dataloader)
        return avg_val_loss

    def save_model(self, model_name: str, is_save_to_disk: bool = False, path_to_save: str = "models/model_1/"):
        # Create the pipeline using the trained modules and save it.
        self.accelerator.wait_for_everyone()
        self.unet = self.accelerator.unwrap_model(self.unet)

        self.pipeline = StableDiffusionPipeline.from_pretrained(
            model_name, text_encoder=self.text_encoder, vae=self.vae, unet=self.unet, revision=None
        )
        self.pipeline.safety_checker = lambda images, clip_input: (images, [False] * len(images))

        if is_save_to_disk:
            self.pipeline.save_pretrained(path_to_save)

        self.accelerator.end_training()

    def load_model(self, path_to_model: str):
        # Load the saved pipeline from the specified directory
        self.pipeline = StableDiffusionPipeline.from_pretrained(path_to_model)

        # Если был отключен safety_checker при сохранении, убедитесь, что он также отключен при загрузке
        self.pipeline.safety_checker = lambda images, clip_input: (images, [False] * len(images))

        return self

    def inference(self, prompt, settings, is_resize=False, target_height=None, target_width=None, device="cuda"):
        """Метод для инференса и ресайза изображений"""

        self.pipeline.to(device)

        # Генерация изображения
        image = self.pipeline(
            prompt,
            guidance_scale=settings.guidance_scale,
            num_inference_steps=settings.num_inference_steps,
            height=settings.height,
            width=settings.width,
            generator=settings.generator,
        ).images[0]

        # Изменение размера с сохранением пропорций
        if is_resize:
            image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)

        return image

    def plot_loss(self, title="Training Loss Over Epochs", xlabel="Epochs", ylabel="Loss"):
        """
        Метод для отрисовки графика loss.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.epoch_losses) + 1), self.epoch_losses, color="blue", label="Train Loss")
        plt.plot(range(1, len(self.val_losses) + 1), self.val_losses, color="orange", label="Val Loss")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        # plt.xticks(range(1, len(self.epoch_losses) + 1))
        plt.legend()

        plt.show()
