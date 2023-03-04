import torch.onnx
from src.processor import Processor

import onnx, logging, os
import numpy as np
from . import utils as U

class Torch2Onnx(Processor):
    def start(self):
        torch_model = self.model
        torch_model.eval()
        torch_model.module
        dummy_input = torch.rand(1,3,144,25,2).to(self.device)
        dummy_out = torch_model(dummy_input)
        print(dummy_out[0])
        print(type(dummy_out[0]))
        # convert pytorch to onnx
        torch.onnx.export(torch_model.module, dummy_input, self.args.onnx_fname, verbose=True, opset_version=12)
        print("success to convert pytorch model to onnx model\n")

        
    def to_numpy(self, tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad() else tensor.cpu().numpy()