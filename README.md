# FastSpeech2-lightning
Training script for Fastspeech2 based on [pytorch lightning](https://pytorch-lightning.readthedocs.io/en/stable/)

# Supported datasets
## JP
* [JSUT](https://sites.google.com/site/shinnosuketakamichi/publication/jsut)

# Run training
## Prerequesties
* Python 3.10 or 3.11 
    * At the time of editing the `torch.compile` is only available on 3.10  
* poetry

# Environment setup
1. Prepare vocoder
    * Vocoder is available on https://github.com/jik876/hifi-gan
1. Setup path
    * checkout config directory for details
1 . Run `poetry install` and enter environment using `poetry shell`

# Training
1. `python src/train.py`

# Acknowledgement
* [ming024's implementation of FastSpeech2](https://github.com/ming024/FastSpeech2). Good portion of the code comes from this repository.
* [jik876's paper and impelementation of HiFi-GAN](https://github.com/jik876/hifi-gan) is used for vocoder.
* [JSUT corpus by Shinnosuke Takamichi](https://sites.google.com/site/shinnosuketakamichi/publication/jsut)
* [Phoneme alignment for JSUT by r9y9](https://github.com/r9y9/jsut-lab)