import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class MLP_Relu_1h(nn.Module):
    """Linear regression model with one hidden layer"""
    def __init__(self, input_dim, n_hid, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, n_hid)
        self.fc2 = nn.Linear(n_hid, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    
class MLP_Relu_2h(nn.Module):
    """Linear regression model with two hidden layer"""
    def __init__(self, input_dim, n_hid, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, n_hid)
        self.fc2 = nn.Linear(n_hid, n_hid)
        self.fc3 = nn.Linear(n_hid, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
class MLP_Silu_1h(nn.Module):
    """Linear regression model with one hidden layer and silu activation"""
    def __init__(self, input_dim, n_hid, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, n_hid)
        self.fc2 = nn.Linear(n_hid, output_dim)

    def forward(self, x):
        x = F.silu(self.fc1(x))
        return self.fc2(x)
    
class MLP_Silu_2h(nn.Module):
    """Linear regression model with two hidden layer and silu activation"""
    def __init__(self, input_dim, n_hid, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, n_hid)
        self.fc2 = nn.Linear(n_hid, n_hid)
        self.fc3 = nn.Linear(n_hid, output_dim)

    def forward(self, x):
        x = F.silu(self.fc1(x))
        x = F.silu(self.fc2(x))
        return self.fc3(x)
    
class MLP_Tanh_1h(nn.Module):
    """Linear regression model with one hidden layer and tanh activation"""
    def __init__(self, input_dim, n_hid, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, n_hid)
        self.fc2 = nn.Linear(n_hid, output_dim)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        return self.fc2(x)

class MLP_Tanh_2h(nn.Module):
    """Linear regression model with two hidden layer and tanh activation"""
    def __init__(self, input_dim, n_hid, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, n_hid)
        self.fc2 = nn.Linear(n_hid, n_hid)
        self.fc3 = nn.Linear(n_hid, output_dim)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        return self.fc3(x)

class Feat_Attn_1h(nn.Module):
    """Attention model with multi-head between-features attention"""
    def __init__(self, input_dim, n_embd, n_head, output_dim):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.embed = nn.Linear(input_dim, n_embd)
        self.pred = nn.Linear(n_embd, output_dim)

    def forward(self, x):
        x = self.embed(x)
        x = rearrange(x, 'B (nh hs) -> B nh hs', nh=self.n_head)
        attention = F.scaled_dot_product_attention(x, x, x)
        attention = rearrange(attention, 'B nh hs -> B (nh hs)')
        return self.pred(attention)

class Feat_Attn_2h(nn.Module):
    """Attention model with multi-head between-features attention"""
    def __init__(self, input_dim, n_embd, n_head, output_dim):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.embed = nn.Linear(input_dim, n_embd)
        self.fc = nn.Linear(n_embd, n_embd)
        self.pred = nn.Linear(n_embd, output_dim)

    def forward(self, x):
        x = self.embed(x)
        x = rearrange(x, 'B (nh hs) -> B nh hs', nh=self.n_head)
        x = F.scaled_dot_product_attention(x, x, x)
        x = rearrange(x, 'B nh hs -> B (nh hs)')
        x = self.fc(x)
        x = rearrange(x, 'B (nh hs) -> B nh hs', nh=self.n_head)
        x = F.scaled_dot_product_attention(x, x, x)
        x = rearrange(x, 'B nh hs -> B (nh hs)')
        return self.pred(x)

class Feat_Attn_3h(nn.Module):
    """Attention model with multi-head between-features attention"""
    def __init__(self, input_dim, n_embd, n_head, output_dim):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        head_dim = n_embd // n_head
        self.embed = nn.Linear(input_dim, n_embd)
        self.fc1 = nn.Linear(head_dim, head_dim)
        self.fc2 = nn.Linear(head_dim, head_dim)
        self.pred = nn.Linear(n_embd, output_dim)

    def forward(self, x):
        x = self.embed(x)
        x = rearrange(x, 'B (nh hs) -> B nh hs', nh=self.n_head)
        x = self.fc1(x)
        x = F.scaled_dot_product_attention(x, x, x)
        x = self.fc2(x)
        x = rearrange(x, 'B nh hs -> B (nh hs)')
        return self.pred(x)