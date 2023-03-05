import torch
import torchaudio
from pathlib import Path
from .preprocessDataset import PreprocessDataset
class JSUTDataset(PreprocessDataset):
    '''
    JSUT dataset class for TTS
    JSUT can be download from https://sites.google.com/site/shinnosuketakamichi/publication/jsut?authuser=0
    Also, alignment is available on https://github.com/r9y9/jsut-lab
    '''
    def __init__(
            self,
            root: Path,
            alignmnet_root: Path,
        ) -> None:
        self.root = root
        self.wav_files = list(self.root.glob('**/*.wav'))
        self.transcript_files = list(self.root.glob('**/transcript_utf8.txt'))
        self.alignment_files = list(self.root.glob('**/*.wav'))
        self.current_wav_index = 0

        self.transcript = dict()
        for transcript_file in self.transcript_files:
            with transcript_file.open('r') as f:
                lines = f.readlines()
                for line in lines:
                    k,v = line.strip().split(':')
                    self.transcript[k] = v
    def parse_alignment_file(self):
        pass

    def __next__(self):
        wav_file_path:Path = self.wav_files[self.current_wav_index]
        transcript:str = self.transcript[wav_file_path.stem]
        alignment = self.pares_alignment_file(wav_file_path.stem)
        return wav_file_path,transcript,alignment


    def __len__(self):
        return len(self.wav_files)


