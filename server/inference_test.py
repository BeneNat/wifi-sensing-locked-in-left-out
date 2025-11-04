import torch
m = torch.jit.load("models/model_tinyml_mlp.pth")
m.eval()
for i in range(5):
    x = torch.randn(1, 28)  # if you have 28 features
    y = m(x)
    print(i, y)
