"""
PyAudio Example: Make a wire between input and output (i.e., record a
few samples and play them back immediately).

This is the callback (non-blocking) version.
"""

import argparse
import json
import yaml
import librosa
import matplotlib.pyplot as plt
# from numba import jit
import numpy as np
import pyaudio
import time
# from librosa.filters import mel as librosa_mel_fn
# from librosa.util import normalize
import torch
# import torchaudio

import logging
import util.dsp as dsp

from agent.base import BaseAgent
from util.config import Config
from util.mytorch import same_seeds


attr_d = {
    "channels": 1,
    "segment_size": 16384,
    "sampling_rate": 22050,
}

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
print(device)

p = pyaudio.PyAudio()
for i in range(p.get_device_count()):
    print(p.get_device_info_by_index(i))

dsp_config_path = 'preprocess.yml'
dsp_config = Config(dsp_config_path)
my_dsp = dsp.Dsp(dsp_config.feat['mel'])
print(my_dsp.config)

model_path = 'again-c4s_100000.pth'
my_config = Config('train_again-c4s.yml')
model_state, step_fn = BaseAgent.build_model(my_config.build, mode='inference', device=device)
model_state = BaseAgent.load_model(model_state, model_path, device=device)

tgt = my_dsp.load_wav('data/wav48/celebs/amitheo.wav')
print('target shape: ', tgt.shape)
tgt_mel = my_dsp.wav2mel(tgt)
print('tgt_mel shape:', tgt_mel.shape)

def callback(in_data, frame_count, time_info, status):
    data = np.frombuffer(in_data, dtype=np.float32)
    data = np.clip(data, -1.0, 1.0)
    src_mel = my_dsp.wav2mel(data)
    print(src_mel.shape)

    with torch.no_grad():
        data = {
            'source': {'mel': src_mel},
            'target': {'mel': tgt_mel},
        }
        meta = step_fn(model_state, data)
        dec = meta['dec']
        y_out = my_dsp.mel2wav(dec)
    
    print(y_out.shape)
    # return (y_out[:attr_d["segment_size"]], pyaudio.paContinue)
    return (y_out[:attr_d["segment_size"]], pyaudio.paContinue)


def main(args=None):
    same_seeds(961998)
    stream = p.open(format=pyaudio.paFloat32,
                    channels=attr_d['channels'],
                    rate=attr_d["sampling_rate"],
                    input=True,
                    output=True,
                    frames_per_buffer=attr_d["segment_size"],
                    stream_callback=callback)

    print("Starting to listen.")
    stream.start_stream()

    while stream.is_active():
        time.sleep(0.1)

    stream.stop_stream()
    stream.close()

    p.terminate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("model_path", type=Path)
    main(**vars(parser.parse_args()))
