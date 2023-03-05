import torch
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

        self.transcript = dict()
        for transcript_file in self.transcript_files:
            with transcript_file.open('r') as f:
                lines = f.readlines()
                for line in lines:
                    k,v = line.strip().split(':')
                    self.transcript[k] = v
        for alignment_file in self.alignment_files:
            with alignment_file.open() as f:
                lines = f.readlines()
                for line in lines:
                    k,v = line.strip().split(':')
        