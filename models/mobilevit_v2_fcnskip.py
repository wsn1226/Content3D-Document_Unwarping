from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU()
    )

def conv_nxn_bn(inp, oup, kernal_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernal_size, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU()
    )

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class PreBatchNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.BatchNorm2d(dim)
        self.fn = fn
    
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

class ConvFFN(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        #print(dim,hidden_dim)
        self.net = nn.Sequential(
            #nn.Linear(dim, hidden_dim),
            #nn.ReLU(),
            nn.Conv2d(dim, hidden_dim, 1, 1, 0, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, dim, 1, 1, 0, bias=True),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        #print(x.shape)
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)

class LinearSelfAttention(nn.Module):
    def __init__(self, dim, dropout=0.):
        super().__init__()
        self.dim=dim
        self.qkv_proj = nn.Conv2d(self.dim, 1 + (2 * self.dim), 1, 1, 0, bias=True)
        self.out_proj = nn.Conv2d(self.dim, self.dim, 1, 1, 0, bias=True)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x):
        #print(x.shape)
        qkv = self.qkv_proj(x)
        query, key, value = torch.split(
            qkv, split_size_or_sections=[1, self.dim, self.dim], dim=1
        )

        context_scores = F.softmax(query, dim=-1)
        context_scores = self.attn_dropout(context_scores)
        # Compute context vector
        # [B, d, P, N] x [B, 1, P, N] -> [B, d, P, N]
        context_vector = key * context_scores
        # [B, d, P, N] --> [B, d, P, 1]
        context_vector = torch.sum(context_vector, dim=-1, keepdim=True)

        # combine context vector with values
        # [B, d, P, N] * [B, d, P, 1] --> [B, d, P, N]
        out = F.relu(value) * context_vector.expand_as(value)
        out = self.out_proj(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))
    
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class LinearAttNFFN(nn.Module):
    def __init__(self, dim, depth, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreBatchNorm(dim, LinearSelfAttention(dim, dropout)),
                PreBatchNorm(dim, ConvFFN(dim, mlp_dim, dropout))
            ]))
    
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class MV2Block(nn.Module):
    def __init__(self, inp, oup, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup
        #print(self.use_res_connect)

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileViTBlock_v2(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = nn.Conv2d(channel, dim, 1, 1, 0, bias=False)

        self.transformer = LinearAttNFFN(dim, depth, mlp_dim, dropout)

        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)

    def unfolding_pytorch(self, feature_map):

        batch_size, in_channels, img_h, img_w = feature_map.shape

        # [B, C, H, W] --> [B, C, P, N]
        patches = F.unfold(
            feature_map,
            kernel_size=(self.ph, self.pw),
            stride=(self.ph, self.pw),
        )
        patches = patches.reshape(
            batch_size, in_channels, self.ph * self.pw, -1
        )

        return patches, (img_h, img_w)

    def folding_pytorch(self, patches, output_size):
        batch_size, in_dim, patch_size, n_patches = patches.shape

        # [B, C, P, N]
        patches = patches.reshape(batch_size, in_dim * patch_size, n_patches)

        feature_map = F.fold(
            patches,
            output_size=output_size,
            kernel_size=(self.ph, self.pw),
            stride=(self.ph, self.pw),
        )

        return feature_map
    
    def forward(self, x):
        y = x.clone()

        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)
        
        # Global representations
        x, output_size = self.unfolding_pytorch(x)
        x = self.transformer(x)
        x = self.folding_pytorch(x,output_size)

        x = self.conv3(x)
        return x
class DecoderConvLayer(nn.Module):
    def __init__(self, input_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.net=nn.Sequential(
        nn.BatchNorm2d(input_channels),
        nn.ConvTranspose2d(in_channels=input_channels,out_channels=out_channels, kernel_size=kernel_size, stride=stride, bias=False),
        nn.ReLU()
        )
    
    def forward(self,x):
        return self.net(x)


class MobileViT_FCN(nn.Module):
    def __init__(self, image_size, dims, channels, num_classes, expansion=4, kernel_size=3, patch_size=(2, 2),input_channels=3,mode=None):
        super().__init__()
        ih, iw = image_size
        ph, pw = patch_size
        assert ih % ph == 0 and iw % pw == 0

        L = [2, 4, 3]

        self.mode=mode
        self.conv1 = conv_nxn_bn(input_channels, channels[0], stride=2)

        self.mv2 = nn.ModuleList([])
        self.mv2.append(MV2Block(channels[0], channels[1], 1, expansion))
        self.mv2.append(MV2Block(channels[1], channels[2], 2, expansion))
        self.mv2.append(MV2Block(channels[2], channels[3], 1, expansion))
        self.mv2.append(MV2Block(channels[2], channels[3], 1, expansion))   # Repeat
        self.mv2.append(MV2Block(channels[3], channels[4], 2, expansion))
        self.mv2.append(MV2Block(channels[5], channels[6], 2, expansion))
        self.mv2.append(MV2Block(channels[7], channels[8], 2, expansion))
        
        self.mvit = nn.ModuleList([])
        self.mvit.append(MobileViTBlock_v2(dims[0], L[0], channels[5], kernel_size, patch_size, int(dims[0])))
        self.mvit.append(MobileViTBlock_v2(dims[1], L[1], channels[7], kernel_size, patch_size, int(dims[1])))
        self.mvit.append(MobileViTBlock_v2(dims[2], L[2], channels[9], kernel_size, patch_size, int(dims[2])))

        self.conv2 = conv_1x1_bn(channels[-2], channels[-1])


        self.pool = nn.AvgPool2d(ih//32, 4)
        #16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640
        self.upconv = nn.ModuleList([])
        self.upconv.append(DecoderConvLayer(640,160,1,1)) #torch.Size([5, 160, 4, 4])
        self.conv3=conv_nxn_bn(320,160)
        self.upconv.append(DecoderConvLayer(160,128,2,2)) #torch.Size([5, 128, 8, 8])
        self.conv4=conv_nxn_bn(256,128)
        self.upconv.append(DecoderConvLayer(128,96,2,2)) #torch.Size([5, 96, 16, 16])
        self.conv5=conv_nxn_bn(192,96)
        self.upconv.append(DecoderConvLayer(96,64,2,2)) #torch.Size([5, 64, 32, 32])
        self.conv6=conv_nxn_bn(128,64)
        self.upconv.append(DecoderConvLayer(64,32,2,2)) #torch.Size([5, 32, 64, 64])
        self.conv7=conv_nxn_bn(64,32)
        self.upconv.append(DecoderConvLayer(32,16,1,1)) #torch.Size([5, 16, 64, 64])
        self.conv8=conv_nxn_bn(32,16)
        self.upconv.append(nn.ConvTranspose2d(16,num_classes,2,2)) #torch.Size([5, 2, 128, 128])
        self.finalactivation=nn.Hardtanh()

        #self.fc = nn.Linear(channels[-1], num_classes, bias=False)

    def forward(self, x):
        skip_connections=[]
        # 5,3,256,256
        x = self.conv1(x)
        # torch.Size([5, 16, 64, 64])
        skip_connections.append(x)
        x = self.mv2[0](x)
        # torch.Size([5, 32, 64, 64])
        skip_connections.append(x)
        x = self.mv2[1](x)
        # torch.Size([5, 64, 32, 32])
        skip_connections.append(x)
        x = self.mv2[2](x)
        x = self.mv2[3](x)      # Repeat
        # torch.Size([5, 64, 16, 16])
        x = self.mv2[4](x)
        # torch.Size([5, 96, 16, 16])
        #print(x.shape)
        x = self.mvit[0](x)
        # torch.Size([5, 96, 16, 16])
        skip_connections.append(x)
        #print(x.shape)

        x = self.mv2[5](x)
        x = self.mvit[1](x)
        #print(x.shape)
        # torch.Size([5, 128, 8, 8])
        skip_connections.append(x)

        x = self.mv2[6](x)
        x = self.mvit[2](x)
        # torch.Size([5, 160, 4 , 4])
        skip_connections.append(x)
        x = self.conv2(x)
        # torch.Size([5, 640, 4 , 4])
        #print(x.shape)
        skip=skip_connections[::-1]

        # FCN Decoder
        x=self.upconv[0](x)
        cat=torch.cat([x,skip[0]],dim=1)
        x=self.conv3(cat)

        x=self.upconv[1](x)
        cat=torch.cat([x,skip[1]],dim=1)
        x=self.conv4(cat)

        x=self.upconv[2](x)
        cat=torch.cat([x,skip[2]],dim=1)
        x=self.conv5(cat)

        x=self.upconv[3](x)
        cat=torch.cat([x,skip[3]],dim=1)
        x=self.conv6(cat)

        x=self.upconv[4](x)
        cat=torch.cat([x,skip[4]],dim=1)
        x=self.conv7(cat)

        x=self.upconv[5](x)
        cat=torch.cat([x,skip[5]],dim=1)
        x=self.conv8(cat)

        x=self.upconv[6](x)
        x=self.finalactivation(x)
        if self.mode=='prob':
            x=torch.clamp(x,min=0,max=1)
        return x


def mobilevit_xxs_skip(img_size=(256,256),input_channels=3,mode=None,num_classes=2):
    dims = [64, 80, 96]
    channels = [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320]
    return MobileViT_FCN(img_size, dims, channels, num_classes=num_classes, expansion=2,input_channels=input_channels,mode=mode)


def mobilevit_xs_skip(img_size=(256,256),input_channels=3,mode=None,num_classes=2):
    dims = [96, 120, 144]
    channels = [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384]
    return MobileViT_FCN(img_size, dims, channels, num_classes=num_classes,input_channels=input_channels,mode=mode)


def mobilevit_v2_s_skip(img_size=(256,256),input_channels=3,mode=None,num_classes=2):
    dims = [144, 256, 288]
    channels = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640]
    return MobileViT_FCN(img_size, dims, channels, num_classes=num_classes,input_channels=input_channels,mode=mode)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    img = torch.randn(5, 4, 128, 128)

    vit = mobilevit_v2_s_skip(input_channels=4,num_classes=2)
    out = vit(img)
    print(out.shape)
    print(torch.min(out),torch.max(out))
    print(count_parameters(vit))


"""     vit = mobilevit_xxs()
    out = vit(img)
    print(out.shape)
    print(count_parameters(vit))

    vit = mobilevit_xs()
    out = vit(img)
    print(out.shape)
    print(count_parameters(vit)) """