import torch
import torchaudio
from pathlib import Path
from .preprocessDataset import PreprocessDataset
class JSUTDataset(torch.utils.data.Dataset):
    '''
    JSUT dataset class for TTS
    JSUT can be download from https://sites.google.com/site/shinnosuketakamichi/publication/jsut?authuser=0
    Also, alignment is available on https://github.com/r9y9/jsut-lab
    '''
    def __init__(
            self,
            root: Path,
            alignmnet_root: Path,
            time_scale_factor:float = 100e-9
        ) -> None:
        self.root = root
        self.wav_files = list(self.root.glob('**/*.wav'))
        self.transcript_files = list(self.root.glob('**/transcript_utf8.txt'))
        alignment_files = list(self.root.glob('**/*.wav'))
        self.current_wav_index = 0
        self.time_scale_factor = time_scale_factor

        self.transcript = dict()
        for transcript_file in self.transcript_files:
            with transcript_file.open('r') as f:
                lines = f.readlines()
                for line in lines:
                    k,v = line.strip().split(':')
                    self.transcript[k] = v

        self.alignment_file_dict= dict()                    
        for alignment_file in alignment_files:
            self.algnment_file_dict[alignment_file.stem] = alignment_file

    def parse_alignment_file(self,file_id:str):
        alignment_file_path:Path = self.alignment_file_dict[file_id]
        phones = []
        starts = []
        ends = []
        with alignment_file_path.open() as f:
            lines = f.readlines()
            for line in lines:
                start, end, full_cntext = line.strip().split(' ')
                starts.append(float(start)*self.time_sclae_factor)
                end.append(float(start)*self.time_scale_factor)
                phones.append(phone)
        return starts, ends, phones
                
    def pp_symbols(self):
        raise NotImplementedError
    def get_item(self):
        wav_file_path:Path = self.wav_files[self.current_wav_index]
        transcript:str = self.transcript[wav_file_path.stem]
        alignment = self.pares_alignment_file(wav_file_path.stem)
        audio = torchaudio.load(wav_file_path)
        return wav_file_path,transcript,alignment

    def __len__(self):
        return len(self.wav_files)
