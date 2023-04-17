import torch
import torch.nn as nn
from .modules import PositionalEncoding, VarianceAdaptor
from .transformer import Encoder, Decoder, PostNet
from .utils import get_mask_from_lengths
from omegaconf import DictConfig
from typing import Dict
from pathlib import Path


class FastSpeech2(nn.Module):
    def __init__(self, cfg: DictConfig, stats_path: Path) -> None:
        super(FastSpeech2, self).__init__()
        self.encoder = Encoder(cfg.encoder)
        self.variance_adaptor = VarianceAdaptor(
            cfg.variance_adaptor, stats_path)

        self.decoder = Decoder(cfg.decoder)
        self.mel_linear = nn.Linear(
            cfg.mel_linear.hidden_size,
            cfg.mel_linear.n_mel_channels
        )
        self.postnet = PostNet()

        self.speaker_emb = None

    def forward(
        self,
        speakers: torch.Tensor,
        texts: torch.Tensor,
        src_lens: torch.Tensor,
        max_src_len: torch.Tensor,
        mels: torch.Tensor = None,
        mel_lens: torch.Tensor = None,
        max_mel_len: torch.Tensor = None,
        pitch_targets: torch.Tensor = None,
        energy_targets: torch.Tensor = None,
        duration_targets: torch.Tensor = None,
        pitch_control: float = 1.0,
        energy_control: float = 1.0,
        duration_control: float = 1.0,
        accents: torch.Tensor = None
    ) -> Dict:
        r"""
        Args:
            speakers: (batch_size)
            texts: (batch_size, max_src_len)
            src_lens: (batch_size)
            max_src_len: (1)
            mels: (batch_size, max_mel_len, n_mel_channels)
            mel_lens: (batch_size)
            max_mel_len: (1)
            pitch_targets: (batch_size, max_mel_len)
            energy_targets: (batch_size, max_mel_len)
            duration_targets: (batch_size, max_mel_len)
            pitch_control: float
            energy_control: float
            duration_control: float
        Returns:
            mel_outputs: (batch_size, max_mel_len, n_mel_channels)
            mel_outputs_postnet: (batch_size, max_mel_len, n_mel_channels)
        """
        src_masks = get_mask_from_lengths(
            src_lens, src_lens.device, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, mel_lens.device, max_mel_len)
            if mel_lens is not None
            else None
        )

        encoder_outputs = self.encoder(texts, src_masks, accents=accents)
        if self.speaker_emb is not None:
            encoder_outputs = encoder_outputs + self.speaker_emb(speakers).unsqueeze(1).expand(
                -1, encoder_outputs.size(1), -1
            )
        (
            variance_adaptor_outputs,
            pitch_predictions,
            energy_predictions,
            log_duration_predictions,
            duration_rounded,
            mel_lens,
            mel_masks
        ) = self.variance_adaptor(
            encoder_outputs,
            src_masks,
            mel_masks,
            max_mel_len,
            pitch_targets,
            energy_targets,
            duration_targets,
            pitch_control,
            energy_control,
            duration_control
        )

        decoder_outputs, mel_masks = self.decoder(
            variance_adaptor_outputs,
            mel_masks
        )

        mel_outputs = self.mel_linear(decoder_outputs)

        postnet_outputs = self.postnet(mel_outputs) + mel_outputs

        return {
            "mel_outputs": mel_outputs,
            "mel_outputs_postnet": postnet_outputs,
            "pitch_predictions": pitch_predictions,
            "energy_predictions": energy_predictions,
            "log_duration_predictions": log_duration_predictions,
            "duration_rounded": duration_rounded,
            "mel_lens": mel_lens,
            "src_masks": src_masks,
            "mel_masks": mel_masks
        }
