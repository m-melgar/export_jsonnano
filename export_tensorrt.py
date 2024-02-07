import io
import numpy as np
import torchvision

from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx
import onnx


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, progress=True)
    model.eval()

    batch_size = 1

    x = torch.randn(batch_size, 3, 720, 1280, requires_grad=True, dtype=torch.float32)


    torch.onnx.export(model,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      "fasterrcnn_resnet50_fpn_in64.onnx",  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=11,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                    'output': {0: 'batch_size'}})

    onnx_model = onnx.load("fasterrcnn_resnet50_fpn_in64.onnx")
    onnx.checker.check_model(onnx_model)

