import torch, torchvision
model = torchvision.models.mobilenet_v2(weights="DEFAULT").eval()
dummy = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy, "mobilenetv2.onnx",
    input_names=["input"], output_names=["output"],
    opset_version=13)
