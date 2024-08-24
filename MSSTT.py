from functools import partial
from torchinfo import summary
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import PatchEmbed,DropPath

from timm.layers.helpers import to_2tuple


class Unfold(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        
        self.kernel_size = kernel_size
        
        weights = torch.eye(kernel_size**2)
        weights = weights.reshape(kernel_size**2, 1, kernel_size, kernel_size)
        self.weights = nn.Parameter(weights, requires_grad=False)
           
        
    def forward(self, x):
        b, c, h, w = x.shape
        x = F.conv2d(x.reshape(b*c, 1, h, w), self.weights, stride=1, padding=self.kernel_size//2)        
        return x.reshape(b, c*1, h*w)

class Fold(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        
        self.kernel_size = kernel_size
        
        weights = torch.eye(kernel_size**2)
        weights = weights.reshape(kernel_size**2, 1, kernel_size, kernel_size)
        self.weights = nn.Parameter(weights, requires_grad=False)
           
        
    def forward(self, x):
        b, _, h, w = x.shape
        x = F.conv_transpose2d(x, self.weights, stride=1, padding=self.kernel_size//2)        
        return x
    



class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, kernel_size=16,  stride=16, padding=0, in_chans=3, embed_dim=768):
        super().__init__()
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding )
        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        # B, C, H, W = x.shape
        x = self.proj(x)
        x = self.norm(x)
        x = x.permute(0,2,3,1)
        return x

class FirstPatchEmbed(nn.Module):
    def __init__(self, kernel_size=3,  stride=1, padding=1, in_chans=3, embed_dim=768):
        super().__init__()      
    def forward(self, x): 
        x = x.permute(0,2,3,1)
        return x

class MSConv(nn.Module):
    def __init__(self, dim, kernel_size=3, stride=1, padding=1,
        **kwargs, ):
        super().__init__()

        self.dim = dim
        self.cnn_in = cnn_in = dim // 2
        self.cnn_dim = cnn_dim = dim - cnn_in
        self.conv1 = nn.Conv2d(cnn_in, dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.proj1 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=False, groups=dim)
        self.mid_gelu1 = nn.GELU()
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=5, stride=stride, padding=2, bias=False, groups=dim)
        self.proj2 = nn.Conv2d(cnn_dim, dim, kernel_size=1, stride=1, padding=0)
        self.mid_gelu2 = nn.GELU()

    def forward(self, x):
        cx = x[:,:self.cnn_in ,:,:].contiguous()
        cx = self.conv1(cx)
        cx = self.proj1(cx)
        cx = self.mid_gelu1(cx)
        
        px = x[:,self.cnn_in:,:,:].contiguous()
        px = self.proj2(px)
        px = self.conv2(px)
        px = self.mid_gelu2(px)
        
        hx = torch.cat((cx, px), dim=1)
        return hx

class MSSTA(nn.Module):
    def __init__(self, dim, num_heads1=8, qkv_bias=False, attn_drop=0., pool_size=1,STA = 1,n_iter=1,qk_scale=None,
        **kwargs, ):

        super().__init__()
        self.STA = STA
        self.n_iter = n_iter       
        self.scale = dim ** - 0.5
        self.unfold = Unfold(1)
        self.fold = Fold(1)
        self.stoken_refine = Attention(dim , num_heads1=1, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=0.,)
        self.dim = dim

    def stoken_forward(self, x):
        '''
           x: (B, C, H, W)
        '''
        B, C, H0, W0 = x.shape
        h, w = self.STA,self.STA
        
        pad_l = pad_t = 0
        pad_r = (w - W0 % w) % w
        pad_b = (h - H0 % h) % h
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (pad_l, pad_r, pad_t, pad_b))#判断填充
            
        _, _, H, W = x.shape
        
        hh, ww = H//h, W//w
        
        stoken_features = F.adaptive_avg_pool2d(x, (hh, ww)) # (B, C, hh, ww)
        
        pixel_features = x.reshape(B, C, hh, h, ww, w).permute(0, 2, 4, 3, 5, 1).reshape(B, hh*ww, h*w, C)#
        
        with torch.no_grad():
            for idx in range(self.n_iter):
                stoken_features = self.unfold(stoken_features) # (B, C*9, hh*ww)
                stoken_features = stoken_features.transpose(1, 2).reshape(B, hh*ww, C, 1)
                affinity_matrix = pixel_features @ stoken_features * self.scale # (B, hh*ww, h*w, 9)
                
                affinity_matrix = affinity_matrix.softmax(-1) # (B, hh*ww, h*w, 9)，
################################################################################
                affinity_matrix_sum = affinity_matrix.sum(2).transpose(1, 2).reshape(B, 1, hh, ww)
               
                affinity_matrix_sum = self.fold(affinity_matrix_sum)
                if idx < self.n_iter - 1:
                    stoken_features = pixel_features.transpose(-1, -2) @ affinity_matrix # (B, hh*ww, C, 9)
                    
                    stoken_features = self.fold(stoken_features.permute(0, 2, 3, 1).reshape(B*C, 1, hh, ww)).reshape(B, C, hh, ww)            
                    
                    stoken_features = stoken_features/(affinity_matrix_sum + 1e-12) # (B, C, hh, ww)
#####################################################################################################################          
        
        stoken_features = pixel_features.transpose(-1, -2) @ affinity_matrix # (B, hh*ww, C, 9),S = ( ( Qt ) TX  
        stoken_features = self.fold(stoken_features.permute(0, 2, 3, 1).reshape(B*C, 1, hh, ww)).reshape(B, C, hh, ww)            
        
        stoken_features = stoken_features/(affinity_matrix_sum.detach() +  1e-12) # (B, C, hh, ww)

        stoken_features = self.stoken_refine(stoken_features)

        
        stoken_features = self.unfold(stoken_features) # (B, C*9, hh*ww)
        stoken_features = stoken_features.transpose(1, 2).reshape(B, hh*ww, C, 1) # (B, hh*ww, C, 9)
       
        pixel_features = stoken_features @ affinity_matrix.transpose(-1, -2) # (B, hh*ww, C, h*w)
        pixel_features = pixel_features.reshape(B, hh, ww, C, h, w).permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)
                     
        if pad_r > 0 or pad_b > 0:
            pixel_features = pixel_features[:, :, :H0, :W0]
        
        return pixel_features
    def direct_forward(self, x):
        B, C, H, W = x.shape
        stoken_features = x
        stoken_features = self.stoken_refine(stoken_features)        
        return stoken_features
        
    def att_fun(self, q, k, v, B, N, C):
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = (attn @ v).transpose(2, 3).reshape(B, C, N)
        return x

    def forward(self, x):
        if self.STA > 1 or self.STA > 1:
            return self.stoken_forward(x)
        else:
            return self.direct_forward(x)
class Attention(nn.Module):
    def __init__(self, dim, window_size=None, num_heads1=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        
        self.dim = dim
        self.num_heads1 = num_heads1
        head_dim = dim // num_heads1
                
        self.window_size = window_size

        self.scale = qk_scale or head_dim ** -0.5
                
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)
        

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
                
        q, k, v = self.qkv(x).reshape(B, self.num_heads1, C // self.num_heads1 *3, N).chunk(3, dim=2) # (B, num_heads, head_dim, N)
        
        attn = (k.transpose(-1, -2) @ q) * self.scale
        
        attn = attn.softmax(dim=-2) # (B, h, N, N)
        attn = self.attn_drop(attn)
        
        x = (v @ attn).reshape(B, C, H, W)
        
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
class Mixer(nn.Module):
    def __init__(self, dim, num_heads1=8, qkv_bias=False, attn_drop=0., proj_drop=0., attention_head=4, pool_size=1,STA1=1, STA2 = 1,STA3 = 1,STA4 = 1,n_iter=1,qk_scale=None, 
        **kwargs, ):
        super().__init__()
        self.STA1 = STA1
        self.STA2 = STA2


        self.n_iter = n_iter       
        self.scale = dim ** - 0.5

    
        self.num_heads1 = num_heads1
        self.head_dim = head_dim = dim // 20


        self.low_dim1 = low_dim1 = attention_head * head_dim *num_heads1


        
        self.high_dim = high_dim = dim - low_dim1*2

        self.high_mixer = MSConv(high_dim)



        self.low_mixer1 = MSSTA(low_dim1, num_heads1=attention_head, qkv_bias=qkv_bias, attn_drop=attn_drop, pool_size=pool_size,STA = STA1,qk_scale=None,n_iter = n_iter,)
        self.low_mixer2 = MSSTA(low_dim1, num_heads1=attention_head, qkv_bias=qkv_bias, attn_drop=attn_drop, pool_size=pool_size,STA = STA2,qk_scale=None,n_iter = n_iter,)

        self.conv_fuse = nn.Conv2d(low_dim1*2+2*high_dim, low_dim1*2+2*high_dim, kernel_size=3, stride=1, padding=1, bias=False, groups=low_dim1*2+2*high_dim)

        self.proj = nn.Conv2d(low_dim1*2+2*high_dim, dim, kernel_size=1, stride=1, padding=0)
        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, x):
        B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2)
        # print(x.shape)
        hx = x[:,:self.high_dim,:,:].contiguous()
        hx = self.high_mixer(hx)
        # print(hx.shape)


        
        lx1 = x[:,self.high_dim:self.high_dim+self.low_dim1,:,:].contiguous()
        lx1 = self.low_mixer1(lx1)


        lx2 = x[:,self.high_dim+self.low_dim1:,:,:].contiguous()
        lx2= self.low_mixer2(lx2)
        # print(lx2.shape)

        x = torch.cat((lx1,lx2,hx), dim=1)
        x = x+self.conv_fuse(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        return x
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., conv_pos=True, downsample=False, kernel_size=5):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features = out_features or in_features
        self.hidden_features = hidden_features = hidden_features or in_features
               
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act1 = act_layer()         
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.conv = ResDWC(hidden_features, 3)
        
    def forward(self, x): 
        x = x.permute(0, 3, 1, 2)   
        x = self.fc1(x)
        x = self.act1(x)
        x = self.drop(x)        
        x = self.conv(x)        
        x = self.fc2(x)               
        x = self.drop(x)
        x = x.permute(0, 2, 3,1) 
        # return (x + x.mean(dim=1, keepdim=True)) * 0.5
        return x 
class ResDWC(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        
        self.dim = dim
        self.kernel_size = kernel_size
        
        self.conv = nn.Conv2d(dim, dim, kernel_size, 1, kernel_size//2, groups=dim)
                
        self.shortcut = nn.Parameter(torch.eye(kernel_size).reshape(1, 1, kernel_size, kernel_size))
        self.shortcut.requires_grad = False
        
    def forward(self, x):
        return F.conv2d(x, self.conv.weight+self.shortcut, self.conv.bias, stride=1, padding=self.kernel_size//2, groups=self.dim) # equal to x + conv(x)
class Block(nn.Module):

    def __init__(self, dim, num_heads1, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, attention_head=1, pool_size=1,STA1=1,STA2=1,STA3 = 1,STA4 = 1,n_iter=1,qk_scale=None,
                 attn=Mixer, 
                 use_layer_scale=False, layer_scale_init_value=1e-5, 
                 ):
        super().__init__()
        
        self.norm1 = norm_layer(dim)
        
        self.attn = attn(dim, num_heads1=num_heads1, qkv_bias=qkv_bias, attn_drop=attn_drop, attention_head=attention_head, pool_size=pool_size,STA1=STA1,STA2 = STA2,STA3 = STA3,STA4 = STA4,n_iter=1,qk_scale=None,)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
        self.use_layer_scale = use_layer_scale
        if self.use_layer_scale:
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.layer_scale_2 * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class MSSTT(nn.Module):
    def __init__(self, img_size=17,  in_chans=200, num_classes=16, embed_dims1=None, depths1=[1, 0, 0, 0],
                 num_heads1=[1,0,0,0], mlp_ratio=4., qkv_bias=True, 
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='',
                 attention_heads=[1]*4 + [3]*6 + [8] * 7 + [10] * 7 + [15] * 6,
                 use_layer_scale=True, layer_scale_init_value=1e-6, 
                 checkpoint_path=None,STA1 = 7,STA2 = 3,STA3= None,STA4= None,n_iter=1,qk_scale=None,
                 **kwargs, 
                 ):
        



        
        super().__init__()
        st2_idx = sum(depths1[:1])
        st3_idx = sum(depths1[:2])
        st4_idx = sum(depths1[:3])
        depth = sum(depths1)
        self.num_classes = num_classes
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.patch_embed = FirstPatchEmbed(in_chans=in_chans, embed_dim=embed_dims1)
        self.num_patches1 = num_patches = img_size 
        self.pos_embed1 = nn.Parameter(torch.zeros(1, num_patches, num_patches, embed_dims1))
        self.blocks1 = nn.Sequential(*[
            Block(
                dim=embed_dims1, num_heads1=num_heads1[0], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer, attention_head=attention_heads[i], pool_size=3,STA1=STA1,STA2=STA2,STA3=STA3,STA4=STA4,n_iter=1,qk_scale=None,)
                # use_layer_scale=use_layer_scale, layer_scale_init_value=layer_scale_init_value, 
                # )
            for i in range(0, st2_idx)])
        self.norm = norm_layer(embed_dims1)
        self.head = nn.Linear(embed_dims1, num_classes) if num_classes > 0 else nn.Identity()
    def forward_features(self, x):
        x = self.patch_embed(x)
        B, H, W, C = x.shape
        x = self.blocks1(x)

        x = x.flatten(1,2)
        x = self.norm(x)
        return x.mean(1)
    
    def forward(self, x):
        x = x.squeeze()
        x = self.forward_features(x)
        x = self.head(x)
        return x
    

    
if __name__ == '__main__':
        n_bands = 200
        patch_size =17
        input = torch.randn(size=(100,n_bands, patch_size, patch_size))
        input = input.cuda()
        print("input shape:", input.shape)
        model = MSSTT( num_classes=9, embed_dims1= n_bands)
        model = model.cuda()
        summary(model, input_size=(2,1,n_bands,patch_size,patch_size),col_names=['num_params','kernel_size','mult_adds','input_size','output_size'],col_width=10,row_settings=['var_names'],depth=4)
        print("output shape:", model(input).shape)