import torch.onnx
from src.processor import Processor

import onnx, logging, os

from . import utils as U

class Torch2Onnx(Processor):
    def start(self):
        torch_model = self.model
        dummy_input = torch.rand(16,3,6,144,25,2).to(self.device)

        # convert pytorch to onnx
        torch.onnx.export(torch_model.module, dummy_input, self.args.onnx_fname, verbose=True, opset_version=12)
        print("success to convert pytorch model to onnx model\n")
