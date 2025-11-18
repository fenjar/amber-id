import torch
import torch.nn as nn
from loss import avatar_contrastive_loss

def dilation_schedule(F):
    if F == 31: return (1,1,1,1,2,2,2,2,4)
    if F == 51: return (1,1,1,1,2,2,2,4,4,4,4)
    if F == 71: return (1,1,1,1,2,2,2,2,2,2,4,4,4,4,4)
    raise ValueError("Unsupported F")

class TemporalIDNet3D(nn.Module):
    def __init__(self, F=71, D=7875, hidden=32, embed_dim=32):
        super().__init__()
        self.F = F
        self.D = D
        dilations = dilation_schedule(F)
        layers = []
        in_ch = 1
        out_ch = hidden
        # first layer kernel=1, dilation=1
        layers += [nn.Conv3d(in_ch, out_ch, kernel_size=(1,1,1), padding=0),
                   nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True)]
        in_ch = out_ch
        # all other layers kernel=(3,1,1) with pre given dilations (temporal only)
        for d in dilations[1:]:
            pad_t = d  # same padding for k=3 along time: pad = dilation
            layers += [
                nn.Conv3d(in_ch, out_ch, kernel_size=(3,1,1),
                          dilation=(d,1,1), padding=(pad_t,0,0)),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True)
            ]
        self.backbone = nn.Sequential(*layers)
        # Aggregation over feature dimensions (H=W=1, only time remains)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool3d((1,1,1)),  # (B, hidden, 1, 1, 1), mittelt über Zeit ebenfalls -> 1 Zeitwert
            nn.Flatten(),                   # (B, hidden)
            # The input features to the linear layer will be (Batch_size, hidden * 1 * (D / 2) * 1)
            # after the pooling layers. Since D is 2016, the input features will be hidden * 1008.
            nn.Linear(hidden, embed_dim)
        )

    def forward_F(self, x_FD):
        # x_FD: (B, F, D)
        x = x_FD.unsqueeze(1).unsqueeze(-1)  # (B,1,F,D,1)
        x = self.backbone(x)                 # (B,hidden,F,1,1) with padding->F stays the same
        z = self.head(x)                     # (B, embed_dim)
        return z

    def forward(self, x):
        # x: (B, T, D), T == F  oder  T == F+4
        B, T, D = x.shape
        assert D == self.D, f"[MODEL] expected D={self.D}, got {D}"
        if T == self.F:
            z = self.forward_F(x)            # (B, E)
            return z.unsqueeze(1)            # (B,1,E)
        elif T == self.F + 4:
            outs = []
            for off in range(5):
                outs.append(self.forward_F(x[:, off:off+self.F, :]))  # (B,E)
            return torch.stack(outs, dim=1)  # (B,5,E)
        else:
            raise ValueError(f"[MODEL] T must be F or F+4, got T={T}, F={self.F}")


# Calculate the number of parameters in the model
def count_parameters(model):
    cnt = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] Number of trainable parameters: {cnt:,}")
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# To check actual memory usage during training, you can use:
#torch.cuda.empty_cache() # Clear cached memory before running
# ... run your training step ...
# print(f"Max memory allocated by PyTorch: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
# print(f"Current memory allocated by PyTorch: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")


# Example usage:
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = TemporalIDNet3D(F=71, D=2016, hidden=128, embed_dim=128).to(device)
# opt = torch.optim.Adam(model.parameters(), lr=1e-4)

# for step, batch in enumerate(loader):
#     x = batch["windows"].to(device)            # (B, F+4, D)
#     z = model(x)                               # (B, 5, E)
#     z_flat = z.reshape(-1, z.size(-1))         # (B*5, E)

#     # baue die passenden Id-Arrays für die 5er-Entfaltung
#     driver_ids = expand_by_5(batch["driver_ids"]).to(device)       # (B*5,)
#     video_ids  = expand_by_5(batch["video_ids"]).to(device)        # (B*5,)
#     is_shuf    = expand_by_5(batch["clip_is_shuffled"]).to(device) # (B*5,) - corrected key
#     pair_idx   = expand_pair_idx_by_5(batch["original_pair_idx"])  # (B*5,)  (Mapping behalten)

#     print(f"Step: {step}")
#     print(f"Shape of x: {x.shape}")
#     print(f"Shape of z: {z.shape}")
#     print(f"Shape of z_flat: {z_flat.shape}")
#     print(f"Shape of driver_ids: {driver_ids.shape}")
#     print(f"Shape of video_ids: {video_ids.shape}")
#     print(f"Shape of is_shuf: {is_shuf.shape}")
#     print(f"Shape of pair_idx: {pair_idx.shape}")


#     loss, stats = avatar_contrastive_loss(
#         embeddings=z_flat,
#         driver_ids=driver_ids,
#         video_ids=video_ids,
#         clip_is_shuffled=is_shuf,
#         original_pair_idx=pair_idx,
#         tau=0.2, wN=1.0, wQ=1.0, wR=0.5
#     )

#     opt.zero_grad()
#     loss.backward()
#     opt.step()