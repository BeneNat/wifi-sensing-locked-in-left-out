import torch

# Model testing
m = torch.jit.load("models/model_tinyml_mlp.pth")
m.eval()
for i in range(5):
    x = torch.randn(1, 28)
    y = m(x)
    print(i, y)
