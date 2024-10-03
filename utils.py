import torch.nn as nn

def show_model_size(model:nn.Module):
    number_of_params = sum(p.numel() for p in model.parameters())
    print(f"number of parameters: {number_of_params/1e6:.6f} M ")
    model_size = number_of_params*4 / 2**20
    print(f"model size: {model_size:.6f} MB")

def init_weights(model:nn.Module):
    for module in model.modules():
        # refer to minGPT initialization
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        # refer to resnet initialization
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)