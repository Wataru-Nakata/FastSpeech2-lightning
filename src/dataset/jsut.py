import torch
import torchaudio
import hydra
from pathlib import Path
from .alignment import Alignment,pp_symbols


class JSUTDataset():
    '''
    JSUT dataset class for TTS
    JSUT can be download from https://sites.google.com/site/shinnosuketakamichi/publication/jsut?authuser=0
    Also, alignment is available on https://github.com/r9y9/jsut-lab
    '''

    def __init__(
        self,
        root: str,
        alignmnet_root: str,
        time_scale_factor: float = 100e-9
    ) -> None:
        self.root:Path = Path(hydra.utils.get_original_cwd()) / Path(root)
        self.alignment_root:Path = Path(hydra.utils.get_original_cwd()) / Path(alignmnet_root)
        self.wav_files = list(self.root.glob('**/*.wav'))
        self.transcript_files = list(self.root.glob('**/transcript_utf8.txt'))
        alignment_files = list(self.alignment_root.glob('**/*.lab'))
        self.current_wav_index = 0
        self.time_scale_factor = time_scale_factor

        self.transcript = dict()
        for transcript_file in self.transcript_files:
            with transcript_file.open('r') as f:
                lines = f.readlines()
                for line in lines:
                    k, v = line.strip().split(':')
                    self.transcript[k] = v

        self.alignment_file_dict = dict()
        for alignment_file in alignment_files:
            self.algnment_file_dict[alignment_file.stem] = alignment_file

    def parse_alignment_file(self, file_id: str):
        alignment_file_path: Path = self.alignment_file_dict[file_id]
        full_contexts = []
        starts = []
        ends = []
        with alignment_file_path.open() as f:
            lines = f.readlines()
            for line in lines:
                start, end, full_cntext = line.strip().split(' ')
                starts.append(float(start)*self.time_sclae_factor)
                end.append(float(start)*self.time_scale_factor)
                full_contexts.append(full_cntext)
        phones, accent = pp_symbols(full_contexts)
        return Alignment(starts, ends, phones,accent)

    def get_item(self):
        wav_file_path: Path = self.wav_files[self.current_wav_index]
        basename = wav_file_path.stem
        transcript: str = self.transcript[wav_file_path.stem]
        alignment = self.parse_alignment_file(wav_file_path.stem)
        return basename,wav_file_path, transcript, alignment,"JSUT"

    def __len__(self):
        return len(self.wav_files)
    
    def __iter__(self):
        return self
    def __next__(self):
        try:
            output = self.get_item()
            self.current_wav_index+=1
            return output
        except IndexError:
            raise StopIteration

