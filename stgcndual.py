"""
stgcndual.py

- Two-phase training: Warmup (Linear head) + Main training (ArcFace head). Classifier head is re-initialized when switching.
- Separate learning rates for warmup (`--lr_warmup`) and main (`--lr_main`). Defaults set to lr_warmup=1e-4, lr_main=5e-5, warmup_epochs=20, total epochs default=200.
- Input scaling and motion channel scaling (args `--input_scale` and `--motion_scale`).
- PK (P x K) sampler support to form batches with multiple identities per batch.
- Mixed-precision using `torch.amp` and recommended GradScaler usage.
- Safe, deterministic dataloader seeding and configurable temporal sampling (crop / pad / uniform sampling).

Example usage:
python stgcndual.py \
  --data_root /path/to/preprocessed_npy \
  --save_dir /path/to/checkpoints \
  --num_epochs 200 --warmup_epochs 20 \
  --batch_p 8 --batch_k 6 --L 64 \
  --lr_warmup 1e-4 --lr_main 5e-5 \
  --input_scale 10.0 --motion_scale 20.0 --amp
"""

import os
import math
import time
import argparse
import random
from glob import glob
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR  # Add this import

# -----------------------------
# Utilities
# -----------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Dataset
# -----------------------------
class LandmarkVideoDataset(Dataset):
    """
    Loads pre-extracted .npy landmark files in a directory structure:
      data_root/
          id_000/
              clip_000.npy  # shape (T_orig, V, 4) -> [x,y,dx,dy]
              ...
          id_001/
              ...

    Performs:
      - temporal sampling to fixed L frames (uniform sampling if T!=L)
      - optional input scaling (input_scale)
      - motion-channel scaling (motion_scale applies to dx,dy)
      - returns tensor shaped (C, L, V) where C == channel_count (usually 4)
    """

    def __init__(self, data_root, L=64, augmentation=True, input_scale=10.0, motion_scale=20.0, seed=42, use_motion=True):
        super().__init__()
        self.data_root = data_root
        self.L = L
        self.augmentation = augmentation
        self.input_scale = input_scale
        self.motion_scale = motion_scale
        self.use_motion = bool(use_motion)

        # Build sample index
        self.samples = []  # list of (label_index, filepath)
        self.id2idx = {}
        ids = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])
        for i, idname in enumerate(ids):
            self.id2idx[idname] = i
            files = sorted(glob(os.path.join(data_root, idname, '*.npy')))
            for f in files:
                self.samples.append((i, f))

        if len(self.samples) == 0:
            raise RuntimeError(f'No samples found under {data_root}')

        self._seed = seed

    def __len__(self):
        return len(self.samples)

    def _temporal_sample(self, data: np.ndarray) -> np.ndarray:
        # data: (T, V, C)
        T, V, C = data.shape
        if T == self.L:
            return data
        if T > self.L:
            # uniform sampling by indices (deterministic or random if augmentation)
            if self.augmentation:
                # random window with length L
                start = random.randint(0, T - self.L)
                return data[start:start + self.L]
            else:
                idxs = np.linspace(0, T - 1, num=self.L).astype(int)
                return data[idxs]
        else:
            # T < L -> pad by repeating last frame (could also interpolate)
            pad = np.repeat(data[-1:, :, :], self.L - T, axis=0)
            return np.concatenate([data, pad], axis=0)

    def __getitem__(self, idx):
        label, fp = self.samples[idx]
        arr = np.load(fp)  # expected shape (T_orig, V, C) with C >= 4 (x,y,dx,dy ...)
        if arr.ndim != 3:
            raise ValueError(f'Wrong sample shape {arr.shape} for {fp}')

        # Handle channel variants depending on use_motion flag:
        # - If file has only x,y and use_motion=True: compute deltas and concat -> (x,y,dx,dy)
        # - If file has only x,y and use_motion=False: keep only (x,y)
        # - If file has >=4 channels: select either first 4 (x,y,dx,dy) or first 2 (x,y) depending on use_motion
        if arr.shape[2] == 2:
            coords = arr  # (T, V, 2)
            if self.use_motion:
                d = np.diff(coords, axis=0, prepend=coords[0:1])
                arr = np.concatenate([coords, d], axis=2)  # (T, V, 4)
            else:
                arr = coords  # (T, V, 2)
        elif arr.shape[2] >= 4:
            if self.use_motion:
                arr = arr[..., :4]
            else:
                arr = arr[..., :2]
        else:
            raise ValueError('Unsupported channel count in npy file')

        # temporal sample/pad to self.L
        arr = self._temporal_sample(arr.astype(np.float32))  # (L, V, 4)

        # apply input scaling (global) and motion scaling
        if self.input_scale is not None and self.input_scale != 1.0:
            arr[..., :2] *= float(self.input_scale)  # scale x,y
        # scale dx,dy only if motion channels are present and enabled
        if self.use_motion and (self.motion_scale is not None and self.motion_scale != 1.0):
            arr[..., 2:4] *= float(self.motion_scale)

        # transpose to (C, L, V)
        arr = arr.transpose(2, 0, 1).astype(np.float32)

        return torch.from_numpy(arr), int(label)


# -----------------------------
# PK Sampler (P identities x K samples) - useful for metric learning and stable ArcFace batches
# -----------------------------
class PKSampler(Sampler):
    def __init__(self, dataset: LandmarkVideoDataset, p: int, k: int):
        super().__init__(dataset)
        self.dataset = dataset
        self.p = p
        self.k = k
        self.id2samples = defaultdict(list)
        for idx, (label, fp) in enumerate(self.dataset.samples):
            self.id2samples[label].append(idx)
        self.ids = list(self.id2samples.keys())

    def __iter__(self):
        # yield flattened indices forming batches of size p*k
        ids = self.ids.copy()
        random.shuffle(ids)
        batch = []
        for id_ in ids:
            samples = self.id2samples[id_]
            if len(samples) >= self.k:
                chosen = random.sample(samples, self.k)
            else:
                chosen = random.choices(samples, k=self.k)
            batch.extend(chosen)
            if len(batch) == self.p * self.k:
                yield from batch
                batch = []
        # leftover pad
        if len(batch) > 0:
            while len(batch) < self.p * self.k:
                id_ = random.choice(self.ids)
                batch.append(random.choice(self.id2samples[id_]))
            yield from batch

    def __len__(self):
        return max(1, len(self.ids) // self.p)


# -----------------------------
# ST-GCN Backbone (AdaptiveGraphConv implementation)
# -----------------------------
class AdaptiveGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        inter = max(8, in_channels // 2)
        self.theta = nn.Conv2d(in_channels, inter, kernel_size=1)
        self.phi = nn.Conv2d(in_channels, inter, kernel_size=1)
        self.g = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # x: (B, C, T, V)
        B, C, T, V = x.shape
        theta = self.theta(x).permute(0, 2, 3, 1).reshape(B * T, V, -1)
        phi = self.phi(x).permute(0, 2, 1, 3).reshape(B * T, -1, V)
        # Attention-like adjacency
        A = torch.matmul(theta, phi)  # (B*T, V, V)
        A = F.softmax(A / math.sqrt(theta.shape[-1]), dim=-1)
        g = self.g(x).permute(0, 2, 3, 1).reshape(B * T, V, -1)
        out = torch.matmul(A, g).reshape(B, T, V, -1).permute(0, 3, 1, 2)
        out = self.bn(out)
        return out


class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1, dropout=0.0):
        super().__init__()
        self.gcn = AdaptiveGraphConv(in_channels, out_channels)
        padding = (kernel_size - 1) // 2
        self.tconv = nn.Conv2d(out_channels, out_channels, kernel_size=(kernel_size, 1), padding=(padding, 0), stride=(stride, 1))
        self.relu = nn.ReLU(inplace=True)
        self.down = None
        if in_channels != out_channels or stride != 1:
            self.down = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)), nn.BatchNorm2d(out_channels))
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        res = x if self.down is None else self.down(x)
        x = self.gcn(x)
        x = self.tconv(x)
        x = self.bn(x)
        # ResNet, only important if in_channels and out_channels donot match or kernel stride isnt 1 step
        x = x + res
        x = self.relu(x)
        x = self.dropout(x)
        return x


class STGCNBackbone(nn.Module):
    def __init__(self, in_channels, num_layers, V, base_channels=64, embedding_size=256, dropout=0.3):
        super().__init__()
        layers = []
        channels = in_channels
        for i in range(num_layers):
            # increase in_channel amount every 3 layers (e.g. 64,64,64,128,128,128,256,...)
            outc = base_channels * (2 ** (i // 3))
            layers.append(STGCNBlock(channels, outc, kernel_size=9, stride=1, dropout=dropout))
            channels = outc
        self.layers = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.embedding = nn.Linear(channels, embedding_size)
        self.bn_emb = nn.BatchNorm1d(embedding_size)

    def forward(self, x):
        # x: (B, C, T, V)
        x = self.layers(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)  # (B, channels)
        emb = self.embedding(x)
        emb = self.bn_emb(emb)
        return emb


# -----------------------------
# ArcFace head
# -----------------------------
class ArcFace(nn.Module):
    def __init__(self, embedding_size, num_classes, s=30.0, m=0.3):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, emb, labels=None):
        emb_norm = F.normalize(emb, p=2, dim=1)
        W = F.normalize(self.weight, p=2, dim=1)
        logits = torch.matmul(emb_norm, W.t())
        if labels is None:
            return logits * self.s
        cosine = logits
        sine = torch.sqrt(torch.clamp(1.0 - cosine ** 2, min=0.0))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi_cond = cosine - self.mm
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        logits_with_margin = torch.where(cosine > self.th, phi, phi_cond)
        output = cosine * (1 - one_hot) + logits_with_margin * one_hot
        output = output * self.s
        return output


# -----------------------------
# Helper forward
# -----------------------------

def classifier_forward(classifier, emb, labels=None):
    if isinstance(classifier, ArcFace):
        return classifier(emb, labels)
    return classifier(emb)


# -----------------------------
# Evaluation
# -----------------------------
@torch.no_grad()
def evaluate(backbone, classifier, dataloader, device):
    backbone.eval()
    classifier.eval()
    correct = 0
    total = 0
    for x, y in dataloader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        emb = backbone(x)
        logits = classifier_forward(classifier, emb, None)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct / total if total > 0 else 0.0


# -----------------------------
# Training loop
# -----------------------------

def train_one_epoch(backbone, classifier, dataloader, optimizer, scaler, device, use_amp=True):
    backbone.train()
    classifier.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for x, y in dataloader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type='cuda', enabled=use_amp):
            emb = backbone(x)
            logits = classifier_forward(classifier, emb, y)
            loss = F.cross_entropy(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)
    return running_loss / total, correct / total


# -----------------------------
# Main
# -----------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', type=str, required=True)
    p.add_argument('--save_dir', type=str, default='./checkpoints')
    p.add_argument('--num_epochs', type=int, default=200)
    p.add_argument('--warmup_epochs', type=int, default=20)
    p.add_argument('--lr_warmup', type=float, default=1e-4)
    p.add_argument('--lr_main', type=float, default=5e-5)
    p.add_argument('--batch_p', type=int, default=8)
    p.add_argument('--batch_k', type=int, default=6)
    p.add_argument('--L', type=int, default=64)
    p.add_argument('--num_layers', type=int, default=9)
    p.add_argument('--base_channels', type=int, default=64)
    p.add_argument('--embedding_size', type=int, default=256)
    p.add_argument('--dropout', type=float, default=0.3)
    p.add_argument('--motion_scale', type=float, default=20.0)
    p.add_argument('--input_scale', type=float, default=10.0)
    p.add_argument('--num_workers', type=int, default=6)
    p.add_argument('--amp', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--no_motion', action='store_true', help='Use positions only (x,y); omit dx,dy channels for ablation')
    p.add_argument('--resume', type=str, default='')
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # dataset + sampler + dataloader
    dataset = LandmarkVideoDataset(args.data_root, L=args.L, augmentation=True, input_scale=args.input_scale, motion_scale=args.motion_scale, seed=args.seed, use_motion=(not args.no_motion))
    num_classes = len(dataset.id2idx)
    sampler = PKSampler(dataset, args.batch_p, args.batch_k)
    batch_size = args.batch_p * args.batch_k
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True)
    val_dataset = LandmarkVideoDataset(args.data_root, L=args.L, augmentation=False, input_scale=args.input_scale, motion_scale=args.motion_scale, seed=args.seed, use_motion=(not args.no_motion))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=max(1, args.num_workers // 2), pin_memory=True)

    # model
    # get first sample to get in_channels and V information
    sample_x, _ = dataset[0]
    in_channels = sample_x.shape[0]  # C
    V = sample_x.shape[2]
    # for xy channels only, in_channels=2, for xydxdy, in_channels=4
    backbone = STGCNBackbone(in_channels=in_channels, num_layers=args.num_layers, V=V, base_channels=args.base_channels, embedding_size=args.embedding_size, dropout=args.dropout).to(device)

    # warmup classifier: linear
    classifier = nn.Linear(args.embedding_size, num_classes).to(device)

    # initial optimizer (warmup lr)
    optimizer = AdamW(list(backbone.parameters()) + list(classifier.parameters()), lr=args.lr_warmup, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.5)  # Halve LR every 50 epochs

    # scaler (recommended API)
    scaler = torch.amp.GradScaler(enabled=args.amp)

    start_epoch = 1
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        backbone.load_state_dict(ck['backbone'])
        try:
            classifier.load_state_dict(ck['classifier'])
        except Exception:
            print('[WARN] Could not load classifier state (shape mismatch). Starting fresh classifier')
        start_epoch = ck.get('epoch', 1) + 1
        optimizer.load_state_dict(ck['opt'])

    best_val = 0.0

    # training loop with warmup -> ArcFace switch
    for epoch in range(start_epoch, args.num_epochs + 1):
        # switch to ArcFace after warmup
        if epoch == args.warmup_epochs + 1:
            print(f"[INFO] Switching to ArcFace head at epoch {epoch} (lr={args.lr_main})")
            classifier = ArcFace(args.embedding_size, num_classes).to(device)
            optimizer = AdamW(list(backbone.parameters()) + list(classifier.parameters()), lr=args.lr_main, weight_decay=1e-4)
            scheduler = StepLR(optimizer, step_size=200, gamma=0.5)  # Re-create scheduler for new optimizer

        t0 = time.time()
        loss, train_acc = train_one_epoch(backbone, classifier, dataloader, optimizer, scaler, device, use_amp=args.amp)
        val_acc = evaluate(backbone, classifier, val_loader, device)
        t1 = time.time()

        print(f'[INFO] Epoch {epoch}/{args.num_epochs} - loss {loss:.4f} acc {train_acc:.4f} val_acc {val_acc:.4f} time {(t1-t0):.1f}s')

        scheduler.step()  # Step the scheduler after each epoch

        # checkpoint
        ck = {'backbone': backbone.state_dict(), 'classifier': classifier.state_dict(), 'epoch': epoch, 'opt': optimizer.state_dict()}

        if (epoch + 1) == 1 or (epoch + 1) % 10 == 0:
            torch.save(ck, os.path.join(args.save_dir, f'ckpt_epoch_{epoch+1:03d}.pt'))
            print(f"[INFO] Saved checkpoint at epoch {epoch+1}")
        if val_acc > best_val:
            best_val = val_acc
            torch.save(ck, os.path.join(args.save_dir, 'best.pt'))

    print('Training completed')


if __name__ == '__main__':
    main()
