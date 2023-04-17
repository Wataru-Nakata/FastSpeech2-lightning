import io
import hydra
import lightning
from lightning.pytorch import loggers
import torch
import torchaudio
from torchvision.transforms import ToTensor
from omegaconf import DictConfig
import lightning
import numpy as np
import PIL
from typing import List
from .utils import get_vocoder
from ..utils import plot_mel


class FS2LightningModule(lightning.LightningModule):
    def __init__(self, cfg: DictConfig, stats_path) -> None:
        super().__init__()
        self.cfg = cfg
        self.model: torch.nn.Module = hydra.utils.instantiate(
            cfg.model, cfg=cfg.model_conf, stats_path=stats_path)
        self.loss: torch.nn.Module = hydra.utils.instantiate(
            cfg.loss, cfg=cfg.loss_conf)
        self.vocoder: torch.nn.Moduel = get_vocoder(cfg.vocoder.vocoder_path)
        self.save_hyperparameters()

        self.pitch_mean = self.model.variance_adaptor.pitch_mean
        self.pitch_std = self.model.variance_adaptor.pitch_std
        self.energy_mean = self.model.variance_adaptor.energy_mean
        self.energy_std = self.model.variance_adaptor.energy_std

    def forward(self, inputs):
        outputs = self.model(**inputs)

        if self.cfg.model_conf.variance_adaptor.normalize:
            inputs["pitch_targets"] = (
                inputs["pitch_targets"] - self.pitch_mean) / self.pitch_std
            inputs["energy_targets"] = (
                inputs["energy_targets"] - self.energy_mean) / self.energy_std
        (
            total_loss,
            mel_loss,
            postnet_mel_loss,
            pitch_loss,
            energy_loss,
            duration_loss
        ) = self.loss(inputs, outputs)
        return (
            total_loss,
            mel_loss,
            postnet_mel_loss,
            pitch_loss,
            energy_loss,
            duration_loss,
            outputs
        )

    def training_step(self, batch, batch_idx):
        (
            total_loss,
            mel_loss,
            postnet_mel_loss,
            pitch_loss,
            energy_loss,
            duration_loss,
            outputs
        ) = self.forward(batch)
        self.log('train_loss/total_loss', total_loss, prog_bar=True)
        self.log('train_loss/mel_loss', mel_loss)
        self.log('train_loss/postnet_mel_loss', postnet_mel_loss)
        self.log('train_loss/pitch_loss', pitch_loss)
        self.log('train_loss/energy_loss', energy_loss)
        self.log('train_loss/duration_loss', duration_loss)
        if batch_idx == 0:
            self.log_mel_audio(
                [
                    [
                        outputs["mel_outputs_postnet"][0].detach()[:batch["mel_lens"][0]].T,
                        self.expand_by_durations(
                            outputs["pitch_predictions"][0].detach().cpu(),
                            batch["duration_targets"][0].detach().cpu()),
                        self.expand_by_durations(
                            outputs['energy_predictions'][0].detach().cpu(),
                            batch["duration_targets"][0].detach().cpu())

                    ],
                    [
                        batch["mels"][0].detach()[:batch["mel_lens"][0]].T,
                        self.expand_by_durations(
                            batch["pitch_targets"][0].detach().cpu(),
                            batch["duration_targets"][0].detach().cpu()),
                        self.expand_by_durations(
                            batch['energy_targets'][0].detach().cpu(),
                            batch["duration_targets"][0].detach().cpu())
                    ]
                ], stage="train", titles=["mel_outputs_postnet", "ground_truth"])

        return total_loss

    def validation_step(self, batch, batch_idx):
        (
            total_loss,
            mel_loss,
            postnet_mel_loss,
            pitch_loss,
            energy_loss,
            duration_loss,
            outputs
        ) = self.forward(batch)
        self.log('val_loss/total_loss', total_loss)
        self.log('val_loss/mel_loss', mel_loss)
        self.log('val_loss/postnet_mel_loss', postnet_mel_loss)
        self.log('val_loss/pitch_loss', pitch_loss)
        self.log('val_loss/energy_loss', energy_loss)
        self.log('val_loss/duration_loss', duration_loss)
        if batch_idx == 0:
            self.log_mel_audio(
                [
                    [
                        outputs["mel_outputs_postnet"][0].detach()[:batch["mel_lens"][0]].T,
                        self.expand_by_durations(
                            outputs["pitch_predictions"][0].detach().cpu(),
                            batch["duration_targets"][0].detach().cpu()),
                        self.expand_by_durations(
                            outputs['energy_predictions'][0].detach().cpu(),
                            batch["duration_targets"][0].detach().cpu())

                    ],
                    [
                        batch["mels"][0].detach()[:batch["mel_lens"][0]].T,
                        self.expand_by_durations(
                            batch["pitch_targets"][0].detach().cpu(),
                            batch["duration_targets"][0].detach().cpu()),
                        self.expand_by_durations(
                            batch['energy_targets'][0].detach().cpu(),
                            batch["duration_targets"][0].detach().cpu())
                    ]
                ], stage="val", titles=["mel_outputs_postnet", "ground_truth"])

        return total_loss

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        outputs = self.model(batch)
        return outputs

    def configure_optimizers(self):
        return hydra.utils.instantiate(self.cfg.optim, params=self.parameters())

    def log_mel_audio(self, mels: List[torch.Tensor], stage, titles):
        image = plot_mel(mels, (0, 1000, self.pitch_mean, self.pitch_std,
                         0, 8), titles)
        audios = [self.synth_wav(mel[0]) for mel in mels]
        for logger in self.loggers:
            match type(logger):
                case loggers.WandbLogger:
                    import wandb
                    logger.log_image(key=f"{stage}/mel",
                                     images=[wandb.Image(image)])
                    for audio, title in zip(audios, titles):
                        wandb.log({f"{stage}/{title}-wav": wandb.Audio(audio, sample_rate=22050)}) 
                case loggers.TensorBoardLogger:
                    buf = io.BytesIO()
                    image.savefig(buf, format="jpeg")
                    buf.seek(0)
                    image = PIL.Image.open(buf)
                    logger.experiment.add_image(
                        f"{stage}/mel", ToTensor()(image), self.global_step)
                    for audio, title in zip(audios, titles):
                        logger.experiment.add_audio(f"{stage}/{title}-wav",audio,self.global_step,22050)

    def expand_by_durations(self, values, durations):
        out = list()
        for value, d in zip(values, durations):
            out += [value] * max(0, int(d))
        return np.array(out)

    def synth_wav(self, mels: torch.Tensor):
        with torch.no_grad():
            wav = self.vocoder(mels)
        return wav.squeeze().cpu().numpy()
