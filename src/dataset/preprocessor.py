import torch
import hydra
import torchaudio
import pathlib
from omegaconf import DictConfig
import pyworld
from scipy.interpolate import interp1d
import numpy as np
import webdataset
from .alignment import Alignment
from .jsut import JSUTDataset


class Preprocessor():
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.dataset = hydra.utils.instantiate(cfg.preprocess.preprocess_dataset)
        self.spec_module = torchaudio.transforms.Spectrogram(
            **cfg.preprocess.stft
        )
        self.mel_scale = torchaudio.transforms.MelScale(
            **cfg.preprocess.mel
        )
        self.sampling_rate = self.cfg.sample_rate
        self.dataset = JSUTDataset(
            root="../../datasets/jsut_ver1.1",
            alignmnet_root="../../datasets/jsut_ver1.1"
        )

    def normalize(self):
        pass

    def process_utterance(
        self,
        basename: str,
        audio_file_path: pathlib.Path,
        transcript: str,
        alignment: Alignment,
        speaker: str,
    ):

        waveform, sample_rate = torchaudio.load(audio_file_path)

        phones, durations, start_time, end_time = self.get_alignment(alignment)
        waveform = torchaudio.functional.resample(
            waveform,
            sample_rate,
            new_freq=self.cfg.sampling_rate
        )

        waveform = waveform[
            int(self.sampling_rate*start_time):int(self.sampling_rate*end_time)
        ]
        pitch, t = pyworld.dio(
            waveform.numpy().astype(np.float64),
            self.sampling_rate,
            frame_period=self.cfg.preprocessing.stft.hop_length
        )

        pitch = pyworld.stonemask(
            waveform.numpy().astype(np.float64),
            pitch,
            t,
            self.sampling_rate
        )
        pitch = pitch[:sum(durations)]

        if np.sum(pitch != 0) <= 1:
            return None

        mel_spec, energy = self.calc_spectrogram(waveform)
        energy = energy[: sum(durations)]

        if self.pitch_phoneme_averaging:
            # perform linear interpolation
            nonzero_ids = np.where(pitch != 0)[0]
            interp_fn = interp1d(
                nonzero_ids,
                pitch[nonzero_ids],
                fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
                bounds_error=False,
            )
            pitch = interp_fn(np.arange(0, len(pitch)))

        if self.cfg.preprocess.pitch_phoneme_averaging:
            # perform linear interpolation
            nonzero_ids = np.where(pitch != 0)[0]
            interp_fn = interp1d(
                nonzero_ids,
                pitch[nonzero_ids],
                fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
                bounds_error=False,
            )
            pitch = interp_fn(np.arange(0, len(pitch)))

            # Phoneme-level average
            # Phoneme-level average
            pos = 0
            for i, d in enumerate(durations):
                if d > 0:
                    pitch[i] = np.mean(pitch[pos: pos + d])
                else:
                    pitch[i] = 0
                pos += d
            pitch = pitch[: len(durations)]

        if self.cfg.preprocess.energy_phoneme_averaging:
            # Phoneme-level average
            pos = 0
            for i, d in enumerate(durations):
                if d > 0:
                    energy[i] = np.mean(energy[pos: pos + d])
                else:
                    energy[i] = 0
                pos += d
            energy = energy[: len(durations)]
        sample = {
            "__key__": basename,
            "phone": " ".join(phones),
            "speaker": speaker,
            "text": transcript,
            "speech.wav": waveform,
            "pitch": pitch,
            "duration": durations,
            "energy": energy,
            "mel": mel_spec
        }
        return sample

    def build_from_path(self):
        sink = webdataset.TarWriter(self.cfg.preprocess.tar_path)
        for basename, wav_file_path, transcript, alignment, speaker in self.dataset:

            sample = self.process_utterance(
                basename,
                wav_file_path,
                transcript,
                alignment,
                speaker
            )
            sink.write(
                sample
            )
        sink.close()


    def get_alignment(self, alignment: Alignment):
        sil_phones = ["sil", "sp", "spn", 'silB', 'silE', '']

        phones = []
        durations = []
        start_time = 0
        end_time = 0
        end_idx = 0
        for out in alignment:
            if alignment.isAccentProvided():
                p, s, e, _ = out
            else:
                p, s, e = out

            # Trim leading silences
            if phones == []:
                if p in sil_phones:
                    continue
                else:
                    start_time = s

            if p not in sil_phones:
                # For ordinary phones
                phones.append(p)
                end_time = e
                end_idx = len(phones)
            else:
                # For silent phones
                phones.append('sp')

            durations.append(
                int(
                    np.round(e * self.sampling_rate / self.hop_length)
                    - np.round(s * self.sampling_rate / self.hop_length)
                )
            )

        # Trim tailing silences
        phones = phones[:end_idx]
        durations = durations[:end_idx]
        assert len(phones) == len(durations)
        return phones, durations, start_time, end_time

    def calc_spectrogram(self, waveform: torch.Tensor):
        magspec = self.spec_module(waveform)
        melspec = self.mel_scale(magspec)
        logmelspec = torch.log(torch.clamp_min(
            melspec, 1.0e-5) * 1.0).to(torch.float32)
        energy = torch.norm(magspec, dim=0)
        return logmelspec.numpy(), energy.numpy()
