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
    parser.add_argument('--vocoder', default=False)

    return parser.parse_args()


def inference_step(model_state, source, target):

    model = model_state['model']
    device = model_state['device']
    model.to(device)
    model.eval()
    
    dec = model.inference(source, target)
    return dec

def _remove_weight_norm(m):
    try:
        # print(f"Weight norm is removed from {m}.")
        torch.nn.utils.remove_weight_norm(m)
    except ValueError:
        # print( "this module didn't have weight norm: ", m)
        return


class VCWrapper(torch.nn.Module):
    def __init__(self, model_state):
        super().__init__()
        self.model = model_state['model']
        self.device = model_state['device']
        self.model.apply(_remove_weight_norm)
        self.model.eval()
        self.model.to(device)


    def forward(self, x, y):
        with torch.no_grad():
            dec = self.model.inference(x, y)
        return dec


class VocoWrapper(torch.nn.Module):
    def __init__(self, vocoder, device):
        super().__init__()
        self.device = device
        self.model = vocoder.mel2wav
        self.model.apply(_remove_weight_norm)
        self.model.eval()
        self.model.to(device)


    def forward(self, x):
        with torch.no_grad():
            dec = self.model(x).view(-1)
        return dec


if __name__ == '__main__':

    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    print(device)

    # build model
    model_path = args.pth
    model_dir = os.path.dirname(model_path)
    model_name = os.path.splitext(model_path)[0]
    my_config = Config(args.config)
    model_state, step_fn = BaseAgent.build_model(my_config.build, mode='inference', device=device)
    model_state = BaseAgent.load_model(model_state, model_path, device=device)

    example_inference_input = torch.rand(1, 80, 128).to(device)
    model = VCWrapper(model_state)

    # Compute inference out for non-scripted model
    dec = model(example_inference_input, example_inference_input)
    dec = dec.detach().cpu().numpy()    
    
    # NOTE: tracing is risky as there are if's and device pinning
    # inputs = {'inference': (example_inference_input, example_inference_input)}
    # module = torch.jit.trace(model, (example_inference_input, example_inference_input))

    # script it!
    scripted_module = torch.jit.script(model)
    inference_out = scripted_module(example_inference_input, example_inference_input)
    print(np.allclose(inference_out.detach().cpu().numpy(), dec, rtol=1e-05, atol=1e-08, equal_nan=False))

    scripted_module.save(model_name + '_' + device + '.pt')
    loaded_model = torch.jit.load(model_name + '_' + device + '.pt')
    loaded_model.to(device)
    loaded_inference_out = loaded_model(example_inference_input, example_inference_input).detach().cpu().numpy()
    print(np.allclose(dec, loaded_inference_out, rtol=1e-05, atol=1e-08, equal_nan=False))

    if args.vocoder:
        # Script vocoder
        melgan = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')
        vocoder = VocoWrapper(melgan, device)
        trace_vocoder = torch.jit.trace(vocoder, inference_out.detach())
        vocoder_out = trace_vocoder(inference_out).detach().cpu().numpy()
        print(vocoder_out)

        with torch.no_grad():
            y = melgan.inverse(inference_out).detach().cpu().numpy().flatten()

        print(np.allclose(vocoder_out, y, rtol=1e-05, atol=1e-08, equal_nan=False))
        melgan_pt_file = os.path.join(model_dir, 'melgan-neurips_' + device + '.pt')
        trace_vocoder.save(melgan_pt_file)
        loaded_vocoder = torch.jit.load(melgan_pt_file)
        loaded_vocoder.to(device)
        loaded_vocoder_out = loaded_vocoder(inference_out.detach()).detach().cpu().numpy()
        print(np.allclose(loaded_vocoder_out, y, rtol=1e-05, atol=1e-08, equal_nan=False))

    
