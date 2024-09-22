import torch
import torch.nn as nn
from einops import rearrange
from timm.models.vision_transformer import Block
from timm.models.layers import DropPath, trunc_normal_
import math
import clip
from thop import profile, clever_format
from txt_utils.model import text_encoder

class DownsampleLayer(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(DownsampleLayer, self).__init__()
        self.Conv_BN_ReLU_2=nn.Sequential(
            nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        self.downsample=nn.Sequential(
            nn.Conv2d(in_channels=out_ch,out_channels=out_ch,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self,x):
        """
        :param x:
        :return: out输出到深层，out_2输入到下一层，
        """
        out=self.Conv_BN_ReLU_2(x)
        out=self.downsample(out)
        return out


class UpSampleLayer(nn.Module):
    def __init__(self,in_ch,out_ch):
        # 512-1024-512
        # 1024-512-256
        # 512-256-128
        # 256-128-64
        super(UpSampleLayer, self).__init__()
        self.Conv_BN_ReLU_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch*2, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(out_ch*2),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_ch*2, out_channels=out_ch*2, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(out_ch*2),
            nn.ReLU()
        )
        self.upsample=nn.Sequential(
            nn.ConvTranspose2d(in_channels=out_ch*2,out_channels=out_ch,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self,x,out):
        '''
        :param x: 输入卷积层
        :param out:与上采样层进行cat
        :return:
        '''
        x_out=self.Conv_BN_ReLU_2(x)
        x_out=self.upsample(x_out)
        cat_out=torch.cat((x_out,out),dim=1)
        return cat_out


class Attention_dec(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # self.task_query = nn.Parameter(torch.randn(1, 512, dim))
        # print(self.task_query.shape)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, task_q):

        B, N, C = x.shape
        # task_q = self.task_query
        # # This is because we fix the task parameters to be of a certain dimension, so with varying batch size, we just stack up the same queries to operate on the entire batch
        # if B > 1:
        #     task_q = task_q.unsqueeze(0).repeat(B, 1, 1, 1)
        #     task_q = task_q.squeeze(1)

        # self.save_task(task_q)

        q = self.q(task_q).reshape(B, task_q.shape[1], self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = 0 - attn
        attn = attn.softmax(dim=-1)
        # print(f'attn:{attn}')
        # print(f'1-attn:{1-attn}')
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv3d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W, C):
        B, N, D = x.shape
        x = x.transpose(1, 2).view(B, D, H, W, C)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W, C):
        x = self.fc1(x)
        x = self.dwconv(x, H, W, C)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block_dec(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_dec(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, q, H, W, C):

        x = x + self.drop_path(self.attn(self.norm1(x), q))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W, C))
        return x

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvLayer, self).__init__()
#         reflection_padding = kernel_size // 2
#         self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
#         out = self.reflection_pad(x)
        out = self.conv2d(x)
        return out


class UpsampleConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
      super(UpsampleConvLayer, self).__init__()
      self.conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=1)

    def forward(self, x):
        out = self.conv2d(x)
        return out


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out) * 0.1
        out = torch.add(out, residual)
        return out

class Decoder(nn.Module):
    def __init__(self, channel=[128, 128, 256, 512]):
        super(Decoder, self).__init__()
        self.up1 = UpsampleConvLayer(in_channels=2 * channel[0], out_channels=channel[0], kernel_size=4, stride=2)
        self.dense1 = nn.Sequential(ResidualBlock(channel[0]))
        self.up2 = UpsampleConvLayer(in_channels=2 * channel[1], out_channels=channel[0], kernel_size=4, stride=2)
        self.dense2 = nn.Sequential(ResidualBlock(channel[0]))
        self.up3 = UpsampleConvLayer(in_channels=2 * channel[2], out_channels=channel[1], kernel_size=4, stride=2)
        self.dense3 = nn.Sequential(ResidualBlock(channel[1]))
        self.up4 = UpsampleConvLayer(in_channels=channel[3], out_channels=channel[2], kernel_size=4, stride=2)
        self.dense4 = nn.Sequential(ResidualBlock(channel[2]))
        self.final_conv = nn.Conv2d(in_channels=channel[0], out_channels=channel[0]//2, kernel_size=1)
    def forward(self, x1, x2, x3, x4):
        x4 = self.up4(x4)
        x4 = self.dense4(x4)
        x3 = torch.cat([x4, x3], dim=1)
        x3 = self.up3(x3)
        x3 = self.dense3(x3)
        x2 = torch.cat([x3, x2], dim=1)
        x2 = self.up2(x2)
        x2 = self.dense2(x2)
        x1 = torch.cat([x2, x1], dim=1)
        x1 = self.up1(x1)
        x1 = self.dense1(x1)
        x = self.final_conv(x1)
        return x

class ChannelLinear(nn.Module):
    def __init__(self, in_dim):
        super(ChannelLinear, self).__init__()
        depths = int(math.log(in_dim, 2))
        modules = []
        for i in range(depths):
            out_dim = in_dim // 2
            modules.append(nn.Linear(in_dim, out_dim))
            modules.append(nn.LeakyReLU())
            in_dim = out_dim
        self.mlp = nn.Sequential(*modules)
    def forward(self, x):
        out = self.mlp(x)
        out = torch.squeeze(out)
        return out

class SpatialLinear(nn.Module):
    def __init__(self, dim, num_classes):
        super(SpatialLinear, self).__init__()

        in_dim1 = dim
        modules1 = []
        for i in range(3):
            out_dim = in_dim1 // 2
            modules1.append(nn.Linear(in_dim1, out_dim))
            modules1.append(nn.LeakyReLU())
            in_dim1 = out_dim
        modules1.append(nn.Linear(in_dim1, num_classes))
        self.mlp1 = nn.Sequential(*modules1)

        in_dim2 = dim
        modules2 = []
        for i in range(3):
            out_dim = in_dim2 // 2
            modules2.append(nn.Linear(in_dim2, out_dim))
            modules2.append(nn.LeakyReLU())
            in_dim2 = out_dim
        modules2.append(nn.Linear(in_dim2, num_classes))
        self.mlp2 = nn.Sequential(*modules2)

        in_dim3 = dim
        modules3 = []
        for i in range(3):
            out_dim = in_dim3 // 2
            modules3.append(nn.Linear(in_dim3, out_dim))
            modules3.append(nn.LeakyReLU())
            in_dim3 = out_dim
        modules3.append(nn.Linear(in_dim3, num_classes))
        self.mlp3 = nn.Sequential(*modules3)

        in_dim4 = dim
        modules4 = []
        for i in range(3):
            out_dim = in_dim4 // 2
            modules4.append(nn.Linear(in_dim4, out_dim))
            modules4.append(nn.LeakyReLU())
            in_dim4 = out_dim
        modules4.append(nn.Linear(in_dim4, num_classes))
        self.mlp4 = nn.Sequential(*modules4)

        self.act = nn.Softmax(dim=-1)

    def forward(self, x1, x2, x3, x4):
        out = self.mlp4(x4) + self.mlp3(x3) + self.mlp2(x2) + self.mlp1(x1)
        out = self.act(out)
        return out

class Restorer(nn.Module):
    def __init__(self, drop_path_rate=0., depths=[3, 4, 6, 3], attn_drop_rate=0., num_heads=[2, 2, 4, 8],
                 mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0., norm_layer=nn.LayerNorm):
        super(Restorer, self).__init__()
        out_channels=[128, 128, 256, 512] #[64, 128, 256, 512, 1024]
        spatial_patch_size=[16, 8, 4, 4]
        channel_patch_size=[16, 16, 32, 16]
        self.d1=DownsampleLayer(3,out_channels[0])#3-64
        self.d2=DownsampleLayer(out_channels[0],out_channels[1])#64-128
        self.d3=DownsampleLayer(out_channels[1],out_channels[2])#128-256
        self.d4=DownsampleLayer(out_channels[2],out_channels[3])#256-512
        self.patch_embed1 = PatchEmbed(spatial_patch_size[0], channel_patch_size[0], 512)
        self.patch_embed2 = PatchEmbed(spatial_patch_size[1], channel_patch_size[1], 512)
        self.patch_embed3 = PatchEmbed(spatial_patch_size[2], channel_patch_size[2], 512)
        self.patch_embed4 = PatchEmbed(spatial_patch_size[3], channel_patch_size[3], 512)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block_dec(
            dim=512, num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=1)
            for i in range(depths[3])])
        # self.mlp1 = Mlp(512, int(mlp_ratios[3] * 512), act_layer=nn.GELU, drop=dpr[cur+depths[3]])


        self.block2 = nn.ModuleList([Block_dec(
            dim=512, num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=1)
            for i in range(depths[2])])
        self.mlp2 = Mlp(512, int(mlp_ratios[2] * 512), act_layer=nn.GELU, drop=dpr[cur + depths[2]])


        self.block3 = nn.ModuleList([Block_dec(
            dim=512, num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=1)
            for i in range(depths[1])])
        self.mlp3 = Mlp(512, int(mlp_ratios[1] * 512), act_layer=nn.GELU, drop=dpr[cur + depths[1]])


        self.block4 = nn.ModuleList([Block_dec(
            dim=512, num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=1)
            for i in range(depths[0])])
        self.mlp4 = Mlp(512, int(mlp_ratios[0] * 512), act_layer=nn.GELU, drop=dpr[cur + depths[0]])

        self.patch_reversion1 = PatchReversion(spatial_patch_size[0], channel_patch_size[0], 4096, channel_size=128, feature_size=128)
        self.patch_reversion2 = PatchReversion(spatial_patch_size[1], channel_patch_size[1], 1024, channel_size=128, feature_size=64)
        self.patch_reversion3 = PatchReversion(spatial_patch_size[2], channel_patch_size[2], 512, channel_size=256, feature_size=32)
        self.patch_reversion4 = PatchReversion(spatial_patch_size[3], channel_patch_size[3], 256, channel_size=512, feature_size=16)

        self.decoder = Decoder(channel=out_channels)
        self.conv_output = ConvLayer(64, 3, kernel_size=3, stride=1, padding=1)
        self.active = nn.Tanh()
        self.txt_encoder = text_encoder()

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def forward(self, x, text):

        # with torch.no_grad():
        txt_fea = self.txt_encoder(text).unsqueeze(dim=1)
        txt_fea = txt_fea.repeat(1,512,1).float()

        out1 = self.d1(x)
        x1 = self.patch_embed1(out1)  # (B, 512, 512)

        out2 = self.d2(out1)
        x2 = self.patch_embed2(out2)  # (B, 512, 512)

        out3 = self.d3(out2)
        x3 = self.patch_embed3(out3)  # (B, 512, 512)

        out4 = self.d4(out3)
        x4 = self.patch_embed4(out4)  # (B, 512, 512)


        for i, blk in enumerate(self.block4):
            x4 = blk(x4, txt_fea, 4, 4, 32)

        x3 = self.mlp4((x3 + x4), 8, 8, 8)

        for i, blk in enumerate(self.block3):
            x3 = blk(x3, txt_fea, 8, 8, 8)

        x2 = self.mlp3((x2 + x3), 8, 8, 8)

        for i, blk in enumerate(self.block2):
            x2 = blk(x2, txt_fea, 8, 8, 8)

        x1 = self.mlp2((x1 + x2), 8, 8, 8)

        for i, blk in enumerate(self.block1):
            x1 = blk(x1, txt_fea, 8, 8, 8)

        x1 = self.patch_reversion1(x1)
        x2 = self.patch_reversion2(x2)
        x3 = self.patch_reversion3(x3)
        x4 = self.patch_reversion4(x4)

        out = self.decoder(x1, x2, x3, x4)
        out = self.conv_output(out)
        out = self.active(out)

        return out


class PatchReversion(nn.Module):
    def __init__(self, spatial_patch_size, channel_patch_size, out_channel, channel_size, feature_size, k=1, dim=512):
        super(PatchReversion, self).__init__()
        self.spatial_patch_size = spatial_patch_size
        self.channel_patch_size = channel_patch_size
        self.nc = int(channel_size // self.channel_patch_size)
        self.np = int(feature_size // self.spatial_patch_size)
        self.mlp = nn.Linear(dim, out_channel)
        self.conv = nn.Conv3d(out_channel, out_channel, kernel_size=k)
    def forward(self, x):
        x = self.mlp(x)
        x = rearrange(x, ' b (nc nh nw) d -> b d nc nh nw ', nc=self.nc, nh=self.np, nw=self.np)
        x = self.conv(x)
        x = rearrange(x, ' b (cp hp wp) nc nh nw -> b (nc cp) (nh hp) (nw wp) ',
                      cp=self.channel_patch_size, hp=self.spatial_patch_size, wp=self.spatial_patch_size)
        return x

class PatchEmbed(nn.Module):
    def __init__(self, spatial_patch_size, channel_patch_size, out_channel, k=1, num_patches=512, embed_dim=512, drop_rate=0.):
        super(PatchEmbed, self).__init__()
        self.out_channel = out_channel
        self.spatial_patch_size = spatial_patch_size
        self.channel_patch_size = channel_patch_size
        self.channel = int(self.channel_patch_size * spatial_patch_size ** 2)
        self.conv = nn.Conv3d(self.channel, self.out_channel, kernel_size=k)
        self.mlp = nn.Linear(self.out_channel, self.out_channel)
        self.norm = nn.LayerNorm(self.out_channel)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
    def forward(self, x):
        x = rearrange(x, " b (nc cp) (nh hp) (nw wp) -> b (cp hp wp) nc nh nw",
                      cp=self.channel_patch_size, hp=self.spatial_patch_size, wp=self.spatial_patch_size)
        x = self.conv(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.mlp(x)
        x = self.norm(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        return x

# model = Restorer().to('cuda')
# inp = torch.rand(size=(1,3,256,256)).to('cuda')
# txt = clip.tokenize(["rain"]).to('cuda')
# # out = model(inp, txt)
# # print(out.shape)
#
# flops, params = profile(model, (inp, txt))
# flops, params = clever_format([flops, params], "%.3f")
# print(params, flops)
