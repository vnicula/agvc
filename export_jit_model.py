import argparse
import torch

from agent.base import BaseAgent
from util.config import Config
from util.mytorch import same_seeds

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pth')
    parser.add_argument('--config')

    return parser.parse_args()


def inference_step(model_state, data):
    meta = {}
    model = model_state['model']
    device = model_state['device']
    model.to(device)
    model.eval()

    source = data['source']['mel']
    target = data['target']['mel']

    source = np2pt(source).to(device)
    target = np2pt(target).to(device)
    
    dec = model.inference(source, target)
    meta = {
        'dec': dec
    }
    return meta

if __name__ == '__main__':

    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    print(device)

    # build model
    model_path = args.pth
    my_config = Config(args.config)
    model_state, step_fn = BaseAgent.build_model(my_config.build, mode='inference', device=device)
    model_state = BaseAgent.load_model(model_state, model_path, device=device)

    model = model_state['model'].eval()
    example_inference_input = torch.rand(1, 80, 128)
    
    # NOTE: tracing is risky as there are if's and device pinning
    # inputs = {'inference': (example_inference_input, example_inference_input)}
    # module = torch.jit.trace_module(model, inputs)

    # script it!
    scripted_module = torch.jit.script(model)

    print(scripted_module(example_inference_input))

    
    
