import torch
from torch import nn
from torchvision.models.vision_transformer import EncoderBlock as ViTLayer

class LayerNorm2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x
    
class LayerNorm3d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim, eps=1e-6)
    
    def forward(self, x):
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = self.ln(x)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x
    
class ConvBlock2D(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False),
            LayerNorm2d(planes),
            nn.GELU(),
            nn.Conv2d(planes, inplanes, kernel_size=3, stride=1, padding=1, groups=groups, bias=False),
            LayerNorm2d(inplanes),
            nn.GELU(),
        )
        self.downsample = downsample if downsample is not None else nn.Identity()

    def forward(self, x):
        return self.layers(x) + self.downsample(x)
    
class ConvBlock3D(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv3d(inplanes, planes, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False),
            LayerNorm3d(planes),
            nn.GELU(),
            nn.Conv3d(planes, inplanes, kernel_size=3, stride=1, padding=1, groups=groups, bias=False),
            LayerNorm3d(inplanes),
            nn.GELU(),
        )
        self.downsample = downsample if downsample is not None else nn.Identity()

    def forward(self, x):
        return self.layers(x) + self.downsample(x)

def prep_Vit2D_grid(x):
    B, C, P, H, W = x.shape
    return x.transpose(1, 2).reshape(B * P, C, H, W).flatten(2).transpose(1, 2)

def recover_Vit2D_grid(x, B, P, C, H, W):
    return x.transpose(1, 2).view(B, P, C, H, W).transpose(1, 2)

def prep_Vit3D_grid(x):
    B, C, P, D, H, W = x.shape
    return x.transpose(1, 2).reshape(B * P, C, D, H, W).flatten(2).transpose(1, 2)

def recover_Vit3D_grid(x, B, P, C, D, H, W):
    return x.transpose(1, 2).view(B, P, C, D, H, W).transpose(1, 2)

def prep_Vit2D(x):
    return x.flatten(2).transpose(1, 2)

def recover_Vit2D(x, B, C, H, W):
    return x.transpose(1, 2).view(B, C, H, W)

def prep_Vit3D(x):
    return x.flatten(2).transpose(1, 2)

def recover_Vit3D(x, B, C, D, H, W):
    return x.transpose(1, 2).view(B, C, D, H, W)

class TransformerBlock2D(nn.Module):
    def __init__(self, vit_args=None):
        super().__init__()
        self.vit = ViTLayer(**vit_args) #if vit_args else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = prep_Vit2D(x)
        x_flat = self.vit(x_flat)
        return recover_Vit2D(x_flat, B, C, H, W)
    
class TransformerBlock3D(nn.Module):
    def __init__(self, vit_args=None):
        super().__init__()
        self.vit = ViTLayer(**vit_args) #if vit_args else nn.Identity()

    def forward(self, x):
        B, C, D, H, W = x.shape
        x_flat = prep_Vit3D(x)
        x_flat = self.vit(x_flat)
        return recover_Vit3D(x_flat, B, C, D, H, W)


class SwinBlock2D(nn.Module):
    def __init__(self, grid_size, shift_size, vit_args=None):
        super().__init__()
        self.grid_size = grid_size
        self.shift_size = shift_size
        self.vit = ViTLayer(**vit_args) #if vit_args else nn.Identity()

    def shift(self, x, H, W):
        shifted_h = (torch.arange(H) + self.shift_size[0]) % H
        shifted_w = (torch.arange(W) + self.shift_size[1]) % W
        return x[:, :, shifted_h, :][:, :, :, shifted_w]

    def reshift(self, x, H, W):
        shifted_h = (torch.arange(H) - self.shift_size[0]) % H
        shifted_w = (torch.arange(W) - self.shift_size[1]) % W
        return x[:, :, shifted_h, :][:, :, :, shifted_w]

    def to_window(self, x, B, C, H, W):
        gh, gw = self.grid_size
        nh, nw = H // gh, W // gw
        parts = nh * nw
        x_ = torch.zeros(B, C, parts, gh, gw, device=x.device, dtype=x.dtype)
        pid = 0
        for h in range(0, H, gh):
            for w in range(0, W, gw):
                x_[:, :, pid] = x[:, :, h:h+gh, w:w+gw]
                pid += 1
        return x_

    def from_window(self, x, B, C, H, W):
        gh, gw = self.grid_size
        x_ = torch.zeros(B, C, H, W, device=x.device, dtype=x.dtype)
        pid = 0
        for h in range(0, H, gh):
            for w in range(0, W, gw):
                x_[:, :, h:h+gh, w:w+gw] = x[:, :, pid]
                pid += 1
        return x_

    def forward(self, x):
        B, C, H, W = x.shape
        assert H % self.grid_size[0] == 0, "H must be divisible by grid height"
        assert W % self.grid_size[1] == 0, "W must be divisible by grid width"

        if any(self.shift_size):
            x = self.shift(x, H, W)

        x_win = self.to_window(x, B, C, H, W)
        x_flat = prep_Vit2D_grid(x_win)
        x_flat = self.vit(x_flat)
        x_win = recover_Vit2D_grid(x_flat, B, x_win.shape[2], C, *self.grid_size)
        x = self.from_window(x_win, B, C, H, W)

        if any(self.shift_size):
            x = self.reshift(x, H, W)

        return x

class SwinBlock3D(nn.Module):
    def __init__(self, grid_size, shift_size, vit_args=None):
        super().__init__()
        self.grid_size = grid_size
        self.shift_size = shift_size
        self.vit = ViTLayer(**vit_args) #if vit_args else nn.Identity()

    def shift(self, x, D, H, W):
        d = (torch.arange(D) + self.shift_size[0]) % D
        h = (torch.arange(H) + self.shift_size[1]) % H
        w = (torch.arange(W) + self.shift_size[2]) % W
        return x[:, :, d, :][:, :, :, h][:, :, :, :, w]

    def reshift(self, x, D, H, W):
        d = (torch.arange(D) - self.shift_size[0]) % D
        h = (torch.arange(H) - self.shift_size[1]) % H
        w = (torch.arange(W) - self.shift_size[2]) % W
        return x[:, :, d, :][:, :, :, h][:, :, :, :, w]

    def to_window(self, x, B, C, D, H, W):
        gd, gh, gw = self.grid_size
        nd, nh, nw = D // gd, H // gh, W // gw
        parts = nd * nh * nw
        x_ = torch.zeros(B, C, parts, gd, gh, gw, device=x.device, dtype=x.dtype)
        pid = 0
        for d in range(0, D, gd):
            for h in range(0, H, gh):
                for w in range(0, W, gw):
                    x_[:, :, pid] = x[:, :, d:d+gd, h:h+gh, w:w+gw]
                    pid += 1
        return x_

    def from_window(self, x, B, C, D, H, W):
        gd, gh, gw = self.grid_size
        x_ = torch.zeros(B, C, D, H, W, device=x.device, dtype=x.dtype)
        pid = 0
        for d in range(0, D, gd):
            for h in range(0, H, gh):
                for w in range(0, W, gw):
                    x_[:, :, d:d+gd, h:h+gh, w:w+gw] = x[:, :, pid]
                    pid += 1
        return x_

    def forward(self, x):
        B, C, D, H, W = x.shape
        assert D % self.grid_size[0] == 0, "D must be divisible by grid depth"
        assert H % self.grid_size[1] == 0, "H must be divisible by grid height"
        assert W % self.grid_size[2] == 0, "W must be divisible by grid width"

        if any(self.shift_size):
            x = self.shift(x, D, H, W)

        x_win = self.to_window(x, B, C, D, H, W)
        x_flat = prep_Vit3D_grid(x_win)
        x_flat = self.vit(x_flat)
        x_win = recover_Vit3D_grid(x_flat, B, x_win.shape[2], C, *self.grid_size)
        x = self.from_window(x_win, B, C, D, H, W)

        if any(self.shift_size):
            x = self.reshift(x, D, H, W)

        return x

class GridBlock2D(nn.Module):
    def __init__(self, grid_size, vit_args=None):
        super().__init__()
        self.grid_size = grid_size
        self.vit = ViTLayer(**vit_args) #if vit_args else nn.Identity()

    def to_grid(self, x, B, C, H, W):
        gh, gw = self.grid_size
        nh, nw = H // gh, W // gw
        parts = nh * nw
        x_ = torch.zeros(B, C, parts, gh, gw, device=x.device, dtype=x.dtype)
        pid = 0
        for h in range(nh):
            for w in range(nw):
                x_[:, :, pid] = x[:, :, h::nh, w::nw]
                pid += 1
        return x_

    def from_grid(self, x, B, C, H, W):
        gh, gw = self.grid_size
        nh, nw = H // gh, W // gw
        x_ = torch.zeros(B, C, H, W, device=x.device, dtype=x.dtype)
        pid = 0
        for h in range(nh):
            for w in range(nw):
                x_[:, :, h::nh, w::nw] = x[:, :, pid]
                pid += 1
        return x_

    def forward(self, x):
        B, C, H, W = x.shape
        assert H % self.grid_size[0] == 0, "H must be divisible by grid height"
        assert W % self.grid_size[1] == 0, "W must be divisible by grid width"
        x_grid = self.to_grid(x, B, C, H, W)
        x_flat = prep_Vit2D_grid(x_grid)
        x_flat = self.vit(x_flat)
        x_grid = recover_Vit2D_grid(x_flat, B, x_grid.shape[2], C, *self.grid_size)
        return self.from_grid(x_grid, B, C, H, W)

class GridBlock3D(nn.Module):
    def __init__(self, grid_size, vit_args=None):
        super().__init__()
        self.grid_size = grid_size
        self.vit = ViTLayer(**vit_args) #if vit_args else nn.Identity()

    def to_grid(self, x, B, C, D, H, W):
        gd, gh, gw = self.grid_size
        nd, nh, nw = D // gd, H // gh, W // gw
        parts = nd * nh * nw
        x_ = torch.zeros(B, C, parts, gd, gh, gw, device=x.device, dtype=x.dtype)
        pid = 0
        for d in range(nd):
            for h in range(nh):
                for w in range(nw):
                    x_[:, :, pid] = x[:, :, d::nd, h::nh, w::nw]
                    pid += 1
        return x_

    def from_grid(self, x, B, C, D, H, W):
        gd, gh, gw = self.grid_size
        nd, nh, nw = D // gd, H // gh, W // gw
        x_ = torch.zeros(B, C, D, H, W, device=x.device, dtype=x.dtype)
        pid = 0
        for d in range(nd):
            for h in range(nh):
                for w in range(nw):
                    x_[:, :, d::nd, h::nh, w::nw] = x[:, :, pid]
                    pid += 1
        return x_

    def forward(self, x):
        B, C, D, H, W = x.shape
        assert D % self.grid_size[0] == 0, "D must be divisible by grid depth"
        assert H % self.grid_size[1] == 0, "H must be divisible by grid height"
        assert W % self.grid_size[2] == 0, "W must be divisible by grid width"
        x_grid = self.to_grid(x, B, C, D, H, W)
        x_flat = prep_Vit3D_grid(x_grid)
        x_flat = self.vit(x_flat)
        x_grid = recover_Vit3D_grid(x_flat, B, x_grid.shape[2], C, *self.grid_size)
        return self.from_grid(x_grid, B, C, D, H, W)

if __name__ == '__main__':
    vit_args = {
        "hidden_dim": 192,
        "num_heads": 12,
        "mlp_dim": 768,
        "dropout": 0.0,
        "attention_dropout": 0.0,
    }
    x3d = torch.randn(2, 192, 12, 16, 20)
    print(torch.allclose(SwinBlock3D((4, 4, 5), (2, 2, 2), vit_args=None)(x3d), x3d, atol=1e-6))
    print(torch.allclose(GridBlock3D((4, 4, 5), vit_args=None)(x3d), x3d, atol=1e-6))

    x2d = torch.randn(2, 192, 16, 16)
    print(torch.allclose(SwinBlock2D((4, 4), (2, 2), vit_args=None)(x2d), x2d, atol=1e-6))
    print(torch.allclose(GridBlock2D((4, 4), vit_args=None)(x2d), x2d, atol=1e-6))
