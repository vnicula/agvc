import argparse
import librosa
import numpy as np
import os
import soundfile as sf
import torch

from pathlib import Path

SR = 22050
NMELS = 80
FMIN = 0
FMAX  = 11025
NFFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024

def load_wav(path):
    y, sr = librosa.load(path, sr=SR)
    y, _ = librosa.effects.trim(y, top_db=20)
    y = np.clip(y, -1.0, 1.0)
    return y

def build_mel_basis():
     return librosa.filters.mel(SR, NFFT,
            fmin=FMIN, fmax=FMAX,
            n_mels=NMELS)

def wav2mel(y, mel_basis):
    D = np.abs(librosa.stft(y, n_fft=NFFT,
        hop_length=HOP_LENGTH, win_length=WIN_LENGTH)**2)
    D = np.sqrt(D)
    S = np.dot(mel_basis, D)
    log_S = np.log10(S)
    return log_S

def main(
    source: Path,
    target: Path,
):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    print(device)
    model = torch.jit.load('steps_30000.pt').to(device)
    vocoder = torch.jit.load('melgan-neurips.pt').to(device)

    src = load_wav(source)
    print('src shape: ', src.shape)
    tgt = load_wav(target)
    print('target shape: ', tgt.shape)

    mbasis = build_mel_basis()
    src_mel = wav2mel(src, mbasis)
    print('src_mel shape:', src_mel.shape)
    tgt_mel = wav2mel(tgt, mbasis)
    print('tgt_mel shape:', tgt_mel.shape)

    source_basename = os.path.basename(source).split('.wav')[0]
    target_basename = os.path.basename(target).split('.wav')[0]
    output_basename = f'{source_basename}_to_{target_basename}'

    src_melt = torch.from_numpy(src_mel[None]).float().to(device)
    tgt_melt = torch.from_numpy(tgt_mel[None]).float().to(device)
    with torch.no_grad():

        # mel = model.inference(src_melt, tgt_melt)
        mel = src_melt
        out = vocoder(mel).cpu().numpy().flatten()

    sf.write(file=output_basename+'.wav', data=out, samplerate=SR)
        # output_plt = os.path.join('plt', output_basename+'.png')
        # source_plt = os.path.join('plt', f'{source_basename}.png')
        # dsp.Dsp.plot_spectrogram(src_mel, source_plt)
        # dsp.Dsp.plot_spectrogram(dec.squeeze().cpu().numpy(), output_plt)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)

    main(**vars(parser.parse_args()))
