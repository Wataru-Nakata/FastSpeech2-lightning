import torch
import hydra
import torchaudio
import pathlib
from omegaconf import DictConfig
import pyworld
from scipy.interpolate import interp1d
import numpy as np
import webdataset
import tqdm
import json
from sklearn.preprocessing import StandardScaler
from .alignment import Alignment
from .jsut import JSUTDataset


class Preprocessor():
    '''
    Preprocess dataset
    '''

    def __init__(self, cfg: DictConfig):
        '''
        Args:
            cfg: hydra config
        '''
        self.cfg = cfg
        self.dataset = hydra.utils.instantiate(
            cfg.preprocess.preprocess_dataset)
        self.spec_module = torchaudio.transforms.Spectrogram(
            **cfg.preprocess.stft
        )
        self.mel_scale = torchaudio.transforms.MelScale(
            **cfg.preprocess.mel
        )
        self.sampling_rate = self.cfg.sample_rate
        pathlib.Path(self.cfg.preprocess.stats_path).parent.mkdir(
            exist_ok=True, parents=True)

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

        phones, durations, start_time, end_time, accent = self.get_alignment(
            alignment)
        waveform = torchaudio.functional.resample(
            waveform,
            sample_rate,
            new_freq=self.sampling_rate
        )[0]  # remove channel dimension only support mono

        waveform = waveform[
            int(self.sampling_rate*start_time):int(self.sampling_rate*end_time)
        ]
        pitch, t = pyworld.dio(
            waveform.numpy().astype(np.float64),
            self.sampling_rate,
            frame_period=self.cfg.preprocess.stft.hop_length / self.sampling_rate * 1000
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
        mel_spec = mel_spec[:, :sum(durations)]
        energy = energy[: sum(durations)]

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
        with open(audio_file_path, mode='rb') as f:
            wav_bytes = f.read()

        sample = {
            "__key__": basename,
            "phone.txt": " ".join(phones),
            "speaker.txt": speaker,
            "text.txt": transcript,
            "speech.wav": wav_bytes,
            "pitch.pth": webdataset.torch_dumps(torch.tensor(pitch)),
            "duration.pth": webdataset.torch_dumps(torch.tensor(durations)),
            "energy.pth": webdataset.torch_dumps(torch.tensor(energy)),
            "mel.pth": webdataset.torch_dumps(mel_spec)
        }
        if accent is not None:
            sample["accent.txt"] = " ".join(accent)
        return sample

    def build_from_path(self):
        train_sink = hydra.utils.instantiate(self.cfg.preprocess.train_tar_sink)
        val_sink = hydra.utils.instantiate(self.cfg.preprocess.val_tar_sink)
        pitch_scaler = StandardScaler()
        energy_scaler = StandardScaler()
        pitch_max_min = {
            "max": -torch.inf,
            "min": torch.inf
        }
        energy_max_min = {
            "max": -torch.inf,
            "min": torch.inf
        }
        speakers = dict()
        vocab = set()
        accent_vocab = set()
        for idx, (basename, wav_file_path, transcript, alignment, speaker) in tqdm.tqdm(enumerate(self.dataset)):

            sample = self.process_utterance(
                basename,
                wav_file_path,
                transcript,
                alignment,
                speaker
            )
            pitch: torch.Tensor = webdataset.torch_loads(sample["pitch.pth"])
            pitch_max_min["max"] = max(
                pitch_max_min["max"],
                pitch.max()
            )
            pitch_max_min["min"] = min(
                pitch_max_min["min"],
                pitch.min()
            )
            pitch_scaler.partial_fit(pitch.numpy().reshape((-1, 1)))
            energy: torch.Tensor = webdataset.torch_loads(sample["energy.pth"])
            energy_max_min["max"] = max(
                energy_max_min["max"],
                energy.max()
            )
            energy_max_min["min"] = min(
                energy_max_min["min"],
                energy.min()
            )
            energy_scaler.partial_fit(energy.numpy().reshape((-1, 1)))
            if idx >= self.cfg.preprocess.val_size:
                train_sink.write(
                    sample
                )
            else:
                val_sink.write(
                    sample
                )
            if speaker not in speakers.keys():
                speakers[speaker] = len(speakers) +1
            [vocab.add(p) for p in sample["phone.txt"].split(" ")]
            if "accent.txt" in sample.keys():
                [accent_vocab.add(p) for p in sample["accent.txt"].split(" ")]

        train_sink.close()
        val_sink.close()
        with open(self.cfg.preprocess.stats_path, mode="w") as f:
            stats = {
                "pitch": [
                    float(pitch_max_min["min"]),
                    float(pitch_max_min["max"]),
                    float(pitch_scaler.mean_[0]),
                    float(pitch_scaler.scale_[0])
                ],
                "energy": [
                    float(energy_max_min["min"]),
                    float(energy_max_min["max"]),
                    float(energy_scaler.mean_[0]),
                    float(energy_scaler.scale_[0])
                ]
            }
            f.write(json.dumps(stats))
        with open(self.cfg.preprocess.speakers_path,mode="w") as f:
            f.write(json.dumps(speakers))
        with open(self.cfg.preprocess.vocab_path,mode="w") as f:
            f.writelines("\n".join(vocab))
        with open(self.cfg.preprocess.accent_vocab_path,mode="w") as f:
            f.writelines("\n".join(accent_vocab))
    def get_alignment(self, alignment: Alignment):
        sil_phones = ["sil", "sp", "spn", 'silB', 'silE', '']

        accents = []
        phones = []
        durations = []
        start_time = 0
        end_time = 0
        end_idx = 0
        for out in alignment:
            if alignment.isAccentProvided():
                p, s, e, a = out
            else:
                p, s, e = out

            # Trim leading silences
            if phones == []:
                if p in sil_phones:
                    continue
                else:
                    start_time = s
            if alignment.isAccentProvided():
                accents.append(a)
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
                    np.round(e * self.sampling_rate /
                             self.cfg.preprocess.stft.hop_length)
                    - np.round(s * self.sampling_rate /
                               self.cfg.preprocess.stft.hop_length)
                )
            )

        # Trim tailing silences
        phones = phones[:end_idx]
        durations = durations[:end_idx]
        assert len(phones) == len(durations)
        if alignment.isAccentProvided():
            accents = accents[:end_idx]
            assert len(accents) == len(phones)
            return phones, durations, start_time, end_time, accents
        else:
            return phones, durations, start_time, end_time, None

    def calc_spectrogram(self, waveform: torch.Tensor):
        magspec = self.spec_module(waveform)
        melspec = self.mel_scale(magspec)
        logmelspec = torch.log(torch.clamp_min(
            melspec, 1.0e-5) * 1.0).to(torch.float32)
        energy = torch.norm(magspec, dim=0)
        return logmelspec, energy.numpy()
