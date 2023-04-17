import torch
import json
import torch.nn.functional as F
import model.hifigan as hifigan
from pathlib import Path


def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0) for i in range(len(input_ele))])

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(
                batch, (0, max_len - batch.size(0)), "constant", 0.0
            )
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0
            )
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded


def get_mask_from_lengths(lengths,device, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(
        0).expand(batch_size, -1).to(device)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)
    return mask

def get_vocoder(vocoder_path:Path):
    vocoder_path = Path(vocoder_path)
    with open(vocoder_path.parent/"config.json", "r") as f:
        config = json.load(f)
    config = hifigan.AttrDict(config)
    vocoder = hifigan.Generator(config)
    ckpt = torch.load(vocoder_path,map_location=torch.device('cpu'))
    vocoder.load_state_dict(ckpt["generator"])
    vocoder.eval()
    vocoder.remove_weight_norm()
    return vocoder
