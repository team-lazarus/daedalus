import torch
from torch import nn
import torch.nn.functional as F

class DaedalusActionPredictor(nn.Module):
    def __init__(self, in_size: int, out_size: int):
        super().__init__()
        
        self.in_keys = ["observation"]
        self.out_keys = ["logits"]

        self.in_size = in_size
        self.out_size = out_size

        self.input_layer = nn.Linear(in_size, 256)
        self.hidden_1 = nn.Linear(256, 512)
        self.hidden_2 = nn.Linear(512, 1024)
        self.output_layer = nn.Linear(1024, out_size)

    def forward(self, tensordict):
        x = tensordict["observation"]
        x = F.tanh(self.input_layer(x))
        x = F.tanh(self.hidden_1(x))
        x = F.tanh(self.hidden_2(x))
        logits = self.output_layer(x)
        # actions = torch.argmax(y, dim=-1)

        tensordict = tensordict.set("logits", logits)
        

        return tensordict