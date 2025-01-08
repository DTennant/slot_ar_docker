from paintmind.stage1.diffloss import DiffLoss
from timm.models.vision_transformer import LayerScale, DropPath, Mlp
from omegaconf import OmegaConf
from paintmind.engine.util import instantiate_from_config
import torch
from torch import nn

class CausalAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def create_causal_mask(self, seq_len):
        mask = torch.triu(
            torch.ones(seq_len, seq_len), diagonal=1
        )
        return mask.unsqueeze(0).unsqueeze(0)
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        mask = self.create_causal_mask(N).to(x.device)
        attn = attn.masked_fill(
            mask == 1, float("-inf")
        )  # apply mask by setting masked positions to -infinity
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class CausalBlock(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CausalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x
    
from einops import repeat

class Mar(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.z_proj = nn.Linear(8, 512)
        self.z_ln = nn.LayerNorm(512)
        self.bos_embed = nn.Parameter(torch.randn(512))

        self.attn_blocks = nn.ModuleList([
            CausalBlock(512, 16, mlp_ratio=2.) for _ in range(24)
        ])

    def forward(self, slots):
        # just use a causal mask
        B, N, C = slots.shape
        slots = self.z_ln(self.z_proj(slots))
        bos = repeat(self.bos_embed.unsqueeze(0), 'n c -> b n c', b=B)
        slots = torch.cat([bos, slots], dim=1)
        
        for _, lyr in enumerate(self.attn_blocks):
            slots = lyr(slots)
        return slots
    
# example_slots = torch.load('example_slot_cache.pth')[:32, :, :].cuda()
# target = example_slots
# model = Mar().cuda()
# loss_fn = DiffLoss(target_channels=8, z_channels=512, depth=12, width=1536, num_sampling_steps="100").cuda()
# # __import__("ipdb").set_trace()
# output = model(example_slots)
# pred = output[:, 1:] # causal model
# # since the pred and output are aligned at all timesteps, we just reshape it.
# pred = pred.reshape(-1, pred.shape[-1])
# target = target.reshape(-1, target.shape[-1])
# loss = loss_fn(target, pred)

class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, *tensors):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].size(0)
    
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

if __name__ == '__main__':
    data = torch.load('example_slot_cache.pth')
    data_shape = data.shape
    data = data.view(data_shape[0], -1)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data = scaler.fit_transform(data.cpu().numpy())
    data = torch.tensor(data).view(*data_shape)
    print(data.max(), data.min())
    
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = Mar().cuda()
    loss_fn = DiffLoss(
        target_channels=8, 
        z_channels=512, 
        depth=12, 
        width=1536, 
        num_sampling_steps="100"
    ).cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer_loss = torch.optim.AdamW(loss_fn.parameters(), lr=1e-4)

    all_losses = []

    for epoch in range(100):
        for idx, batch in enumerate(dataloader):
            slots = batch[0].cuda()
            target = slots
            output = model(slots)
            pred = output[:, 1:] # causal model
            # since the pred and output are aligned at all timesteps, we just reshape it.
            pred = pred.reshape(-1, pred.shape[-1])
            target = target.reshape(-1, target.shape[-1])
            loss = loss_fn(target, pred)

            optimizer.zero_grad()
            optimizer_loss.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer_loss.step()
            
            all_losses.append(loss.item())
            if idx % 10 == 0:
                print(f'Epoch: {epoch}, Iteration: {idx}, Loss: {loss.item()}')

    torch.save(model.state_dict(), 'mar_model.pth')
    torch.save(loss_fn.state_dict(), 'mar_loss_fn.pth')
    torch.save(scaler, 'mar_scaler.pth')


