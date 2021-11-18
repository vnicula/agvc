import argparse
import librosa
import matplotlib.pyplot as plt
import os
import torch

from pathlib import Path

import logging
import util.dsp as dsp

from agent.base import BaseAgent
from util.config import Config
from util.mytorch import same_seeds
# from util.vocoder import get_vocoder

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(asctime)s | %(filename)s | %(message)s',\
     datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

def save_spect(save_path, spec):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(spec.T, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)
    fig.savefig(save_path)

    return fig


def main(
    model_path: Path,
    source: Path,
    target: Path,
    output: Path,
):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    print(device)
    # model = torch.jit.load(model_path).to(device)
    # vocoder = torch.jit.load(vocoder_path).to(device)
    # model = load_autovc(model_path).to(device)

    # vocoder = get_vocoder(device)
    # print(vocoder)
    same_seeds(961998)

    dsp_config_path = 'preprocess.yml'
    dsp_config = Config(dsp_config_path)
    my_dsp = dsp.Dsp(dsp_config.feat['mel'])
    print(my_dsp.config)
    src = my_dsp.load_wav(source)
    print('src shape: ', src.shape)
    tgt = my_dsp.load_wav(target)
    print('target shape: ', tgt.shape)

    src_mel = my_dsp.wav2mel(src)
    print('src_mel shape:', src_mel.shape)
    tgt_mel = my_dsp.wav2mel(tgt)
    print('tgt_mel shape:', tgt_mel.shape)

    my_config = Config('train_again-c4s.yml')
    model_state, step_fn = BaseAgent.build_model(my_config.build, mode='inference', device=device)
    model_state = BaseAgent.load_model(model_state, model_path, device=device)
    # print(model_state)
    with torch.no_grad():
        data = {
            'source': {'mel': src_mel},
            'target': {'mel': tgt_mel},
        }
        meta = step_fn(model_state, data)
        dec = meta['dec']
        my_dsp.mel2wav(dec, output)

        source_basename = os.path.basename(source).split('.wav')[0]
        target_basename = os.path.basename(target).split('.wav')[0]
        output_basename = f'{source_basename}_to_{target_basename}'
        output_plt = os.path.join('plt', output_basename+'.png')
        source_plt = os.path.join('plt', f'{source_basename}.png')
        dsp.Dsp.plot_spectrogram(src_mel, source_plt)
        dsp.Dsp.plot_spectrogram(dec.squeeze().cpu().numpy(), output_plt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("output", type=Path)
    main(**vars(parser.parse_args()))
