import torch.onnx
from src.processor import Processor

import onnx, logging, os
import onnxruntime
import numpy as np
from . import utils as U

class Torch2Onnx(Processor):
    def start(self):
        torch_model = self.model
        torch_model.eval()
        dummy_input = torch.rand(1,3,144,25,2).to(self.device)
        dummy_out = torch_model(dummy_input)
        # convert pytorch to onnx
        
        # torch.onnx.export(torch_model.module, dummy_input, self.args.onnx_fname, verbose=True, opset_version=12)
        # print("success to convert pytorch model to onnx model\n")


        onnx_model = onnx.load("convert/data/out.onnx")
        onnx.checker.check_model(onnx_model)
        ort_session = onnxruntime.InferenceSession("convert/data/out.onnx")
        ort_inputs = {ort_session.get_inputs()[0].name: self.to_numpy(dummy_input)}
        ort_outs = ort_session.run(None, ort_inputs)
        
        print("pytorch 모델 추론 결과:" + dummy_out[0])
        print("onnx 모델 추론 결과:" + ort_outs[0])
        # ONNX 런타임과 PyTorch에서 연산된 결과값 비교
        np.testing.assert_allclose(self.to_numpy(dummy_out[0]), ort_outs[0], rtol=1e-03, atol=1e-05)

        print("Exported model has been tested with ONNXRuntime, and the result looks good!")
        
    def to_numpy(self, tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()