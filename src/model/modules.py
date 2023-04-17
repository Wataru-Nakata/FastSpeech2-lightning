import hydra
import json
from omegaconf import DictConfig
import torch
import pathlib
import torch.nn as nn
from .utils import get_mask_from_lengths, pad
from collections import OrderedDict

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class VarianceAdaptor(nn.Module):
    """ Variance Adaptor """

    def __init__(self, cfg:DictConfig,stats_path:pathlib.Path):
        super(VarianceAdaptor, self).__init__()
        self.duration_predictor = VariancePredictor(cfg.duration_predictor)
        self.length_regulator = LengthRegulator()
        self.pitch_predictor = VariancePredictor(cfg.pitch_predictor)
        self.energy_predictor = VariancePredictor(cfg.energy_predictor)

        self.pitch_feature_level = "phoneme_level" if cfg.pitch_phoneme_averaging else "frame_level"
        self.energy_feature_level = "phoneme_level" if  cfg.energy_phoneme_averaging else "frame_level"
        assert self.pitch_feature_level in ["phoneme_level", "frame_level"]
        assert self.energy_feature_level in ["phoneme_level", "frame_level"]

        pitch_quantization = cfg.pitch_quantization
        energy_quantization = cfg.energy_quantization
        n_bins = cfg.n_bins
        assert pitch_quantization in ["linear", "log"]
        assert energy_quantization in ["linear", "log"]
        with stats_path.open() as f:
            stats = json.load(f)
        self.stats = stats
        pitch_min, pitch_max, pitch_mean, pitch_std = stats["pitch"]
        energy_min, energy_max, energy_mean, energy_std = stats["energy"]
        self.cfg = cfg

        if self.cfg.normalize:
            pitch_min = (pitch_min - pitch_mean)/pitch_std
            pitch_max = (pitch_max - pitch_mean)/pitch_std
            energy_max = (energy_max - energy_mean)/energy_std
            energy_min = (energy_min - energy_mean)/energy_std
            self.pitch_mean = pitch_mean
            self.pitch_std = pitch_std
            self.energy_mean = energy_mean
            self.energy_std = energy_std

        if pitch_quantization == "log":
            self.pitch_bins = nn.Parameter(
                torch.exp(
                    torch.linspace(torch.log(pitch_min), torch.log(pitch_max), n_bins - 1)
                ),
                requires_grad=False,
            )
        else:
            self.pitch_bins = nn.Parameter(
                torch.linspace(pitch_min, pitch_max, n_bins - 1),
                requires_grad=False,
            )
        if energy_quantization == "log":
            self.energy_bins = nn.Parameter(
                torch.exp(
                    torch.linspace(torch.log(energy_min), torch.log(energy_max), n_bins - 1)
                ),
                requires_grad=False,
            )
        else:
            self.energy_bins = nn.Parameter(
                torch.linspace(energy_min, energy_max, n_bins - 1),
                requires_grad=False,
            )

        self.pitch_embedding = nn.Embedding(
            n_bins, cfg.pitch_embedding_dim
        )
        self.energy_embedding = nn.Embedding(
            n_bins, cfg.energy_embedding_dim
        )

    def get_pitch_embedding(self, x, target, mask, control):
        prediction = self.pitch_predictor(x, mask)
        if target is not None:
            embedding = self.pitch_embedding(torch.bucketize(target, self.pitch_bins))
        else:
            prediction = prediction * control
            embedding = self.pitch_embedding(
                torch.bucketize(prediction, self.pitch_bins)
            )
        return prediction, embedding

    def get_energy_embedding(self, x, target, mask, control):
        prediction = self.energy_predictor(x, mask)
        if target is not None:
            embedding = self.energy_embedding(torch.bucketize(target, self.energy_bins))
        else:
            prediction = prediction * control
            embedding = self.energy_embedding(
                torch.bucketize(prediction, self.energy_bins)
            )
        return prediction, embedding

    def forward(
        self,
        x,
        src_mask,
        mel_mask=None,
        max_len=None,
        pitch_target=None,
        energy_target=None,
        duration_target=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):
        if self.cfg.normalize:
            pitch_target = (pitch_target - self.pitch_mean)/self.pitch_std
            energy_target = (energy_target - self.energy_mean)/self.energy_std

        log_duration_prediction = self.duration_predictor(x, src_mask)
        if self.pitch_feature_level == "phoneme_level":
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(
                x, pitch_target, src_mask, p_control
            )
            x = x + pitch_embedding
        if self.energy_feature_level == "phoneme_level":
            energy_prediction, energy_embedding = self.get_energy_embedding(
                x, energy_target, src_mask, p_control
            )
            x = x + energy_embedding

        if duration_target is not None:
            x, mel_len = self.length_regulator(x, duration_target, max_len)
            duration_rounded = duration_target
        else:
            duration_rounded = torch.clamp(
                (torch.round(torch.exp(log_duration_prediction) - 1) * d_control),
                min=0,
            )
            x, mel_len = self.length_regulator(x, duration_rounded, max_len)
            mel_mask = get_mask_from_lengths(mel_len,mel_len.device)

        if self.pitch_feature_level == "frame_level":
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(
                x, pitch_target, mel_mask, p_control
            )
            x = x + pitch_embedding
        if self.energy_feature_level == "frame_level":
            energy_prediction, energy_embedding = self.get_energy_embedding(
                x, energy_target, mel_mask, p_control
            )
            x = x + energy_embedding

        return (
            x,
            pitch_prediction,
            energy_prediction,
            log_duration_prediction,
            duration_rounded,
            mel_len,
            mel_mask,
        )


class LengthRegulator(nn.Module):
    """ Length Regulator """

    def __init__(self):
        super(LengthRegulator, self).__init__()

    def LR(self, x:torch.Tensor, duration, max_len):
        output = list()
        mel_len = list()
        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            mel_len.append(expanded.shape[0])

        if max_len is not None:
            output = pad(output, max_len)
        else:
            output = pad(output)

        return output, torch.LongTensor(mel_len).to(x.device)

    def expand(self, batch, predicted):
        out = list()

        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            out.append(vec.expand(max(int(expand_size), 0), -1))
        out = torch.cat(out, 0)

        return out

    def forward(self, x, duration, max_len):
        output, mel_len = self.LR(x, duration, max_len)
        return output, mel_len


class VariancePredictor(nn.Module):
    """ Duration, Pitch and Energy Predictor """

    def __init__(self, cfg:DictConfig):
        super(VariancePredictor, self).__init__()

        self.input_size = cfg.input_size
        self.filter_size = cfg.filter_size
        self.kernel = cfg.kernel_size
        self.conv_output_size = cfg.output_conv_size
        self.dropout = cfg.dropout

        self.conv_layer = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1d_1",
                        Conv(
                            self.input_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=(self.kernel - 1) // 2,
                        ),
                    ),
                    ("relu_1", nn.ReLU()),
                    ("layer_norm_1", nn.LayerNorm(self.filter_size)),
                    ("dropout_1", nn.Dropout(self.dropout)),
                    (
                        "conv1d_2",
                        Conv(
                            self.filter_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=1,
                        ),
                    ),
                    ("relu_2", nn.ReLU()),
                    ("layer_norm_2", nn.LayerNorm(self.filter_size)),
                    ("dropout_2", nn.Dropout(self.dropout)),
                ]
            )
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 1)

    def forward(self, encoder_output, mask):
        out = self.conv_layer(encoder_output)
        out = self.linear_layer(out)
        out = out.squeeze(-1)

        if mask is not None:
            out = out.masked_fill(mask, 0.0)

        return out


class Conv(nn.Module):
    """
    Convolution Module
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        w_init="linear",
    ):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        x = x.contiguous().transpose(1, 2)

        return x