from my_codes.my_blocks import *

class Local2GlobalUNet3Dv3(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, channel=48, num_layers=[2,2,2,3]):
        super(Local2GlobalUNet3Dv3, self).__init__()
        
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, channel, kernel_size=5, stride=1, padding=2),
            LayerNorm3d(channel),
            nn.GELU(),
            nn.Conv3d(channel, channel, kernel_size=5, stride=1, padding=2),
        )
        conv_encoder = [ConvBlock3D(channel*2,channel*4) for _ in range(num_layers[0])]
        self.encoder1 = nn.Sequential(*conv_encoder)
        vit_args = {
        "hidden_dim": channel*4,
        "num_heads": 12,
        "mlp_dim": channel*16,
        "dropout": 0.0,
        "attention_dropout": 0.0,
        }
        swin_encoder = [SwinBlock3D((6,8,8),(3,4,4) if i%2==0 else (0,0,0),vit_args=vit_args) for i in range(num_layers[1])]
        self.encoder2 = nn.Sequential(*swin_encoder)
        grid_encoder = [GridBlock3D((3,4,4),vit_args=vit_args) for _ in range(num_layers[2])]
        self.encoder3 = nn.Sequential(*grid_encoder)
        vit_args = {
        "hidden_dim": channel*8,
        "num_heads": 12,
        "mlp_dim": channel*32,
        "dropout": 0.0,
        "attention_dropout": 0.0,
        }
        bottleneck = [TransformerBlock3D(vit_args=vit_args) for _ in range(num_layers[3])]
        self.bottleneck = nn.Sequential(*bottleneck)
        downsamplers = [
            nn.Conv3d(channel, channel*2, kernel_size=2, stride=2, padding=0),
            nn.Conv3d(channel*2, channel*4, kernel_size=2, stride=2, padding=0),
            nn.Conv3d(channel*4, channel*4, kernel_size=2, stride=2, padding=0),
            nn.Conv3d(channel*4, channel*8, kernel_size=2, stride=2, padding=0),
        ]
        self.downsamplers = nn.ModuleList(downsamplers)
        upsamplers = [
                nn.ConvTranspose3d(channel*8, channel*4, kernel_size=2, stride=2, padding=0),
                nn.ConvTranspose3d(channel*4, channel*4, kernel_size=2, stride=2, padding=0),
                nn.ConvTranspose3d(channel*4, channel*2, kernel_size=2, stride=2, padding=0),
                nn.ConvTranspose3d(channel*2, channel, kernel_size=2, stride=2, padding=0),
        ]
        self.upsamplers = nn.ModuleList(upsamplers)
        conv_decoder = [nn.Conv3d(channel*4, channel*2, kernel_size=1, stride=1, padding=0)]+[ConvBlock3D(channel*2,channel*4) for _ in range(num_layers[0])]
        self.decoder1 = nn.Sequential(*conv_decoder)
        vit_args = {
        "hidden_dim": channel*4,
        "num_heads": 12,
        "mlp_dim": channel*16,
        "dropout": 0.0,
        "attention_dropout": 0.0,
        }
        swin_decoder = [nn.Conv3d(channel*8, channel*4, kernel_size=1, stride=1, padding=0)]+[SwinBlock3D((6,8,8),(3,4,4) if i%2==0 else (0,0,0),vit_args=vit_args) for i in range(num_layers[1])]
        self.decoder2 = nn.Sequential(*swin_decoder)
        grid_decoder = [nn.Conv3d(channel*8, channel*4, kernel_size=1, stride=1, padding=0)]+[GridBlock3D((3,4,4),vit_args=vit_args) for _ in range(num_layers[2])]
        self.decoder3 = nn.Sequential(*grid_decoder)
        self.out = nn.Sequential(
            nn.Conv3d(channel*2, channel, kernel_size=3, stride=1, padding=1),
            LayerNorm3d(channel),
            nn.GELU(),
            nn.Conv3d(channel, out_channels, kernel_size=1, stride=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = x.permute(0,1,4,2,3)
        x0 = self.stem(x)
        x1 = self.encoder1(self.downsamplers[0](x0))
        x2 = self.encoder2(self.downsamplers[1](x1))
        x3 = self.encoder3(self.downsamplers[2](x2))
        x = self.bottleneck(self.downsamplers[3](x3))

        x = self.decoder3(torch.cat([x3,self.upsamplers[0](x)],dim=1))
        x = self.decoder2(torch.cat([x2,self.upsamplers[1](x)],dim=1))
        x = self.decoder1(torch.cat([x1,self.upsamplers[2](x)],dim=1))
        x = self.out(torch.cat([x0,self.upsamplers[3](x)],dim=1))
        x = x.permute(0,1,3,4,2)
        return x
    

if __name__ == "__main__":
    model = Local2GlobalUNet3Dv3().cuda()
    x = torch.randn(1,2,128,128,96).cuda()
    y = model(x)
    print(y.shape)