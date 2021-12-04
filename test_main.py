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
import soundfile as sf
import time
import torch

import logging
import util.dsp as dsp

from agent.base import BaseAgent
from util.config import Config
from util.mytorch import same_seeds


attr_d = {
    "channels": 1,
    "segment_size": 8191,
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
dsp_config.feat['mel']['trim'] = 30
my_dsp = dsp.Dsp(dsp_config.feat['mel'])
print(my_dsp.config)

# model_path = 'again-c4s_100000.pth'
model_path = 'trained_models/steps_100000.pth'
my_config = Config('train_again-c4s.yml')
model_state, step_fn = BaseAgent.build_model(my_config.build, mode='inference', device=device)
model_state = BaseAgent.load_model(model_state, model_path, device=device)

tgt = my_dsp.load_wav('data/wav48/celebs/lady_gaga.wav')
print('target shape: ', tgt.shape)
tgt_mel = my_dsp.wav2mel(tgt)
print('tgt_mel shape:', tgt_mel.shape)
# warm up vocoder
_ = my_dsp.mel2wav(torch.from_numpy(tgt_mel[None]).float())

def callback(in_data, frame_count, time_info, status):
    data = np.frombuffer(in_data, dtype=np.float32)
    np.clip(data, -1.0, 1.0)
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


def main_inp(args=None):
    same_seeds(961998)
    stream = p.open(format=pyaudio.paFloat32,
                    channels=attr_d['channels'],
                    rate=attr_d["sampling_rate"],
                    input_device_index=1,
                    input=True,
                    # output=True,
                    frames_per_buffer=attr_d["segment_size"])

    out = np.empty((0,), dtype=np.float32)

    print("Starting to listen.")
    data = stream.read(attr_d["segment_size"])
    for i in range(0, 10):
        raw_data = np.frombuffer(data, dtype=np.float32)
        print('data range: %.3f, %.3f' % (raw_data.min(), raw_data.max()))
        # TODO: tune all these plus segment_size
        vad_data, index_vad = librosa.effects.trim(raw_data, top_db=my_dsp.config['trim'], ref=0.5, frame_length=512, hop_length=128)
        lead_silence = raw_data[:index_vad[0]]
        trail_silence = raw_data[index_vad[1]:]
        print(index_vad)
        y_out = vad_data
        # data = np.zeros_like(vad_data)
        if index_vad[1] - index_vad[0] >= 2048:
            # np.clip(vad_data, -1.0, 1.0)
            src_mel = my_dsp.wav2mel(vad_data)
            print(src_mel.shape)

            with torch.no_grad():
                data_dict = {
                    'source': {'mel': src_mel},
                    'target': {'mel': tgt_mel},
                }
                meta = step_fn(model_state, data_dict)
                dec = meta['dec']
                y_out = my_dsp.mel2wav(dec)
                # y_out = my_dsp.mel2wav(src_mel)
        
        print('shape out: ', y_out.shape)
        reco = np.concatenate([lead_silence, y_out[:len(vad_data)], trail_silence], axis=0)
        print('reco shape: ', reco.shape)
        out = np.append(out, reco[:attr_d["segment_size"]], axis=0)
        print(out.shape)
        # stream.write(y_out.tobytes(), attr_d["segment_size"])
        
        # NOTE: need exception_on_overflow = True
        data = stream.read(attr_d["segment_size"], exception_on_overflow=True)
    
    sf.write(file='10secs.wav', data=out, samplerate=attr_d["sampling_rate"])
    stream.stop_stream()
    stream.close()

    p.terminate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("model_path", type=Path)
    main_inp(**vars(parser.parse_args()))
