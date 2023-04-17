import webdataset as wds
import lightning
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
import torch
import json


class FS2DataModule(lightning.LightningDataModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # construct vocab
        self.vocab = dict()
        with Path(self.cfg.vocab_path).open() as f:
            lines = f.readlines()
        for idx, l in enumerate(lines):
            self.vocab[l.strip()] = idx

        # accent vocab building
        self.accent_vocab = dict()
        with Path(self.cfg.accent_vocab_path).open() as f:
            lines = f.readlines()
        for idx, l in enumerate(lines):
            self.accent_vocab[l.strip()] = idx

        # constract speaker
        with Path(self.cfg.speakers_path).open() as f:
            self.speakers = json.load(f)

    def setup(self, stage: str):
        self.train_dataset = wds.WebDataset(self.cfg.train_dataset_path).shuffle(1000).decode(
            wds.torch_audio
        )
        self.val_dataset = wds.WebDataset(self.cfg.val_dataset_path).shuffle(1000).decode(
            wds.torch_audio
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.cfg.train_batch_size, collate_fn=self.collate_fn,num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.cfg.val_batch_size, collate_fn=self.collate_fn,num_workers=0)

    def collate_fn(self, batch):
        outputs = dict()
        outputs["mels"] = pad_sequence(
            [b["mel.pth"].T for b in batch], batch_first=True)
        outputs["pitch_targets"] = pad_sequence(
            [b["pitch.pth"] for b in batch], batch_first=True).type(torch.FloatTensor)
        outputs["energy_targets"] = pad_sequence(
            [b["energy.pth"] for b in batch], batch_first=True)
        outputs["duration_targets"] = pad_sequence(
            [b["duration.pth"] for b in batch], batch_first=True)
        outputs["texts"] = pad_sequence(
            [torch.tensor([self.vocab[p] for p in b["phone.txt"].split(" ")])for b in batch], batch_first=True)
        outputs["src_lens"] = torch.tensor(
            [len(b["phone.txt"].split(" ")) for b in batch])

        outputs["speakers"] = torch.tensor([self.speakers[b["speaker.txt"]] for b in batch])
        outputs["max_src_len"] = outputs["src_lens"].max()
        outputs["mel_lens"] = torch.tensor(
            [b["mel.pth"].size(1) for b in batch])
        outputs["max_mel_len"] = outputs["mel_lens"].max()
        if "accent.txt" in batch[0].keys():
            outputs["accents"] = pad_sequence(
                [torch.tensor([self.accent_vocab[p] for p in b["accent.txt"].split(" ")])for b in batch], batch_first=True)
        return outputs
