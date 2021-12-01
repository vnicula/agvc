import argparse
import numpy as np
import torch
import os

from agent.base import BaseAgent
from util.config import Config
from util.mytorch import same_seeds

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pth')
    parser.add_argument('--config')

    return parser.parse_args()


def inference_step(model_state, source, target):

    model = model_state['model']
    device = model_state['device']
    model.to(device)
    model.eval()
    
    dec = model.inference(source, target)
    return dec

if __name__ == '__main__':

    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    print(device)

    # build model
    model_path = args.pth
    model_name = os.path.splitext(model_path)[0]
    my_config = Config(args.config)
    model_state, step_fn = BaseAgent.build_model(my_config.build, mode='inference', device=device)
    model_state = BaseAgent.load_model(model_state, model_path, device=device)

    example_inference_input = torch.rand(1, 80, 128)
    # Compute inference out for non-scripted model
    with torch.no_grad():
        dec = inference_step(model_state, example_inference_input, example_inference_input)
    dec = dec.detach().cpu().numpy()    

    model = model_state['model'].eval()
    
    # NOTE: tracing is risky as there are if's and device pinning
    # inputs = {'inference': (example_inference_input, example_inference_input)}
    # module = torch.jit.trace_module(model, inputs)

    # script it!
    scripted_module = torch.jit.script(model)
    inference_out = scripted_module.inference(example_inference_input, example_inference_input)
    print(np.allclose(inference_out.detach().cpu().numpy(), dec, rtol=1e-05, atol=1e-08, equal_nan=False))

    scripted_module.save(model_name + '.pt')
    loaded_model = torch.jit.load(model_name + '.pt')
    loaded_model.to(device)
    loaded_inference_out = loaded_model.inference(example_inference_input, example_inference_input).detach().cpu().numpy()
    print(np.allclose(dec, loaded_inference_out, rtol=1e-05, atol=1e-08, equal_nan=False))

    # Script vocoder
    vocoder = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')
    voco_net = vocoder.mel2wav.eval()
    trace_vocoder = torch.jit.trace(voco_net, inference_out.detach())
    vocoder_out = trace_vocoder(inference_out.detach()).detach().cpu().numpy().flatten()
    print(vocoder_out)

    with torch.no_grad():
        y = vocoder.inverse(inference_out).detach().cpu().numpy().flatten()
    print(np.allclose(vocoder_out, y, rtol=1e-05, atol=1e-08, equal_nan=False))
    trace_vocoder.save('melgan-neurips.pt')
    loaded_vocoder = torch.jit.load('melgan-neurips.pt')
    loaded_vocoder.to(device)
    loaded_vocoder_out = loaded_vocoder(inference_out.detach()).detach().cpu().numpy()
    print(np.allclose(loaded_vocoder_out, y, rtol=1e-05, atol=1e-08, equal_nan=False))


    
    
