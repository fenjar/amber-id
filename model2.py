"""
stgcn_arcface_starter.py

PyTorch starter script for "Assessing Privacy of anonymized Video" project.

Includes:
- Dataset that loads Mediapipe FaceMesh landmark .npy files (expected shape T x V x 4 or T x V x 3)
- Preprocessing: centering, interocular normalization, temporal cropping/padding, simple augmentations
- PKSampler (P identities x K samples per batch)
- Simple adaptive ST-GCN backbone (graph adjacency learned adaptively) suitable for landmark sequences
- ArcFace (additive angular margin) classification head
- Training loop with AMP (torch.cuda.amp), checkpointing, validation, and DDP hints in comments

Usage notes:
- This is a starter scaffold. You need to run Mediapipe extraction separately and save per-video .npy files with landmarks.
  A recommended format: landmarks.npy shaped (T, V, 4) where columns are [x, y, z, visibility] or (T, V, 3).
- Set DATA_ROOT to point at a directory structured like:
    DATA_ROOT/
      id_000/
        video_000.npy
        video_001.npy
      id_001/
        ...
  where folder names correspond to integer class labels (or map them in a labels.csv and adapt Dataset).
- This script intentionally keeps the ST-GCN implementation compact and uses adaptive spatial graph.

Requirements:
- python >=3.8
- torch
- numpy
- optional: mediapipe for landmark extraction (not required to run training if landmarks already extracted)

Run (single GPU):
    python stgcn_arcface_starter.py --data_root /path/to/data --epochs 80 --batch_p 16 --batch_k 4

Distributed (DDP) notes:
- For multi-GPU on a node, launch via torch.distributed.launch or torchrun and uncomment DDP parts.

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

# -----------------------------
# Config & Utilities
# -----------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', type=str, required=True, help='Root dir with per-ID subfolders containing .npy landmark files')
    p.add_argument('--num_epochs', type=int, default=80)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--batch_p', type=int, default=16, help='P identities per batch')
    p.add_argument('--batch_k', type=int, default=4, help='K samples per identity in batch')
    p.add_argument('--num_workers', type=int, default=6)
    p.add_argument('--L', type=int, default=96, help='Temporal window length (frames)')
    p.add_argument('--embedding_size', type=int, default=256)
    p.add_argument('--dropout', type=float, default=0.3)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--save_dir', type=str, default='./checkpoints')
    p.add_argument('--amp', action='store_true', help='Enable mixed precision (recommended)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--use_arcface', action='store_true', default=False)
    p.add_argument('--s', type=float, default=30.0, help='ArcFace scale')
    p.add_argument('--m', type=float, default=0.3, help='ArcFace margin')
    p.add_argument('--resume', type=str, default='')
    return p.parse_args()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Dataset
# -----------------------------

class LandmarkVideoDataset(Dataset):
    """
    Expects a directory with subfolders per identity, each containing .npy files with landmarks.
    Landmark file format: (T, V, 4) or (T, V, 3) where last dim is [x,y,z,(visibility)].

    Preprocessing pipeline included:
      - center on nose tip (or landmark index 1 fallback)
      - scale by interocular distance
      - temporal crop/pad to length L
      - compute deltas optionally
    """
    def __init__(self, data_root, L=96, augmentation=True, compute_deltas=True):
        super().__init__()
        self.data_root = data_root
        self.L = L
        self.augmentation = augmentation
        self.compute_deltas = compute_deltas

        # build index: list of (id_label, filepath)
        self.samples = []
        self.id2idx = {}
        ids = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])
        for i, idname in enumerate(ids):
            self.id2idx[idname] = i
            files = glob(os.path.join(data_root, idname, '*.npy'))
            for f in files:
                self.samples.append((i, f))

        print(f'Found {len(ids)} identities, {len(self.samples)} samples total')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        label, fp = self.samples[idx]
        data = np.load(fp)  # expected shape (T, V, 3/4)

        # basic sanity
        if data.ndim != 3:
            raise ValueError(f'Unexpected landmark shape {data.shape} in {fp}')

        T, V, C = data.shape

        # ensure we have visibility channel; if not, append ones
        if C == 3:
            vis = np.ones((T, V, 1), dtype=data.dtype)
            data = np.concatenate([data, vis], axis=2)
            C = 4

        # Temporal crop / pad to self.L
        if T >= self.L:
            if self.augmentation:
                start = np.random.randint(0, T - self.L + 1)
            else:
                start = max(0, (T - self.L) // 2)
            data = data[start:start + self.L]
        else:
            # pad by repeating last frame
            pad_len = self.L - T
            pad = np.repeat(data[-1:,...], pad_len, axis=0)
            data = np.concatenate([data, pad], axis=0)

        # normalize coordinates per video
        data = self._normalize_frames(data)  # returns (L, V, 4)

        # optionally compute deltas
        if self.compute_deltas:
            coords = data[..., :3]  # (L, V, 3)
            d1 = np.concatenate([coords[1:] - coords[:-1], np.zeros((1, V, 3), dtype=coords.dtype)], axis=0)
            # second derivative
            d2 = np.concatenate([d1[1:] - d1[:-1], np.zeros((1, V, 3), dtype=coords.dtype)], axis=0)
            # combine: x,y,z, d1x,d1y,d1z, visibility
            data = np.concatenate([coords, d1, d2, data[..., 3:4]], axis=2)  # (L, V, 7)

        # transpose to (C, L, V) for model (channels first)
        data = data.transpose(2, 0, 1).astype(np.float32)

        return torch.from_numpy(data), int(label)

    def _normalize_frames(self, data):
        # data: (T, V, 4)
        # choose nose tip as anchor if available (mediapipe approx index 1), else landmark 1
        # If your landmarks use a different indexing, adjust here.
        nose_idx = 1 if data.shape[1] > 1 else 0
        # compute centroid anchor per frame
        anchor = data[:, nose_idx, :3]  # (T, 3)
        coords = data[..., :3] - anchor[:, None, :]

        # compute interocular distance: distance between left-eye and right-eye center
        # common mediapipe indices: left eye cluster ~ 33-133 area, right eye ~ 263-362 — we'll approximate with two points if available
        left_eye_idx = 33 if data.shape[1] > 33 else 0
        right_eye_idx = 263 if data.shape[1] > 263 else (data.shape[1] - 1)
        left = data[:, left_eye_idx, :3]
        right = data[:, right_eye_idx, :3]
        iod = np.linalg.norm(left - right, axis=1)  # (T,)
        iod_mean = iod.mean() if iod.mean() > 1e-6 else 1.0

        coords = coords / iod_mean

        # keep visibility channel as-is
        vis = data[..., 3:4]
        out = np.concatenate([coords, vis], axis=2)
        return out


# -----------------------------
# PK Sampler for P x K batches
# -----------------------------

class PKSampler(Sampler):
    """
    Samples batch containing P identities and K samples each.
    Expects dataset.samples to be list of (label, filepath).
    """
    def __init__(self, data_source, p, k):
        super().__init__(data_source)
        self.data_source = data_source
        self.p = p
        self.k = k
        self.id2samples = defaultdict(list)
        for idx, (label, fp) in enumerate(data_source.samples):
            self.id2samples[label].append(idx)
        self.ids = list(self.id2samples.keys())

    def __iter__(self):
        # yield indices grouped into batches
        id_list = self.ids.copy()
        random.shuffle(id_list)
        batch = []
        for id_ in id_list:
            samples = self.id2samples[id_]
            if len(samples) >= self.k:
                chosen = random.sample(samples, self.k)
            else:
                chosen = random.choices(samples, k=self.k)
            batch.extend(chosen)
            if len(batch) == self.p * self.k:
                yield from batch
                batch = []
        # handle leftover
        if len(batch) > 0:
            # pad with random
            while len(batch) < self.p * self.k:
                label = random.choice(self.ids)
                sample = random.choice(self.id2samples[label])
                batch.append(sample)
            yield from batch

    def __len__(self):
        # number of batches per epoch approximated by floor(total_ids / p)
        return max(1, len(self.ids) // self.p)


# -----------------------------
# ST-GCN backbone (compact)
# -----------------------------

class AdaptiveGraphConv(nn.Module):
    """
    Graph convolution where adjacency is learned adaptively from node embeddings.
    This avoids hardcoding face mesh topology here and remains flexible.
    Input shape: (B, C, T, V)
    Output shape: (B, out_channels, T, V)
    """
    def __init__(self, in_channels, out_channels, V, coff=1):
        super().__init__()
        self.V = V
        inter_channels = max(8, in_channels // 2)
        self.theta = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.phi = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.g = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # x: (B, C, T, V)
        B, C, T, V = x.shape
        theta = self.theta(x).permute(0,2,3,1).reshape(B*T, V, -1)  # (B*T, V, inter)
        phi = self.phi(x).permute(0,2,1,3).reshape(B*T, -1, V)       # (B*T, inter, V)
        A = torch.matmul(theta, phi)  # (B*T, V, V)
        A = self.softmax(A / math.sqrt(theta.shape[-1]))
        g = self.g(x).permute(0,2,3,1).reshape(B*T, V, -1)  # (B*T, V, out)
        out = torch.matmul(A, g)  # (B*T, V, out)
        out = out.reshape(B, T, V, -1).permute(0,3,1,2).contiguous()  # (B, out, T, V)
        out = self.bn(out)
        return out


class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, V, kernel_size=9, stride=1, dropout=0.0):
        super().__init__()
        self.gcn = AdaptiveGraphConv(in_channels, out_channels, V)
        padding = (kernel_size - 1) // 2
        self.tconv = nn.Conv2d(out_channels, out_channels, kernel_size=(kernel_size,1), padding=(padding,0), stride=(stride,1))
        self.relu = nn.ReLU(inplace=True)
        self.down = None
        if in_channels != out_channels or stride != 1:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride,1)),
                nn.BatchNorm2d(out_channels)
            )
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout) if dropout>0 else nn.Identity()

    def forward(self, x):
        # x: (B, C, T, V)
        res = x if self.down is None else self.down(x)
        x = self.gcn(x)
        x = self.tconv(x)
        x = self.bn(x)
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
            outc = base_channels * (2 ** (i//3))  # increase every 3 layers
            layers.append(STGCNBlock(channels, outc, V, kernel_size=9, stride=1, dropout=dropout))
            channels = outc
        self.layers = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1,1))  # pool over time and nodes -> (B, C, 1,1)
        self.embedding = nn.Linear(channels, embedding_size)
        self.bn_emb = nn.BatchNorm1d(embedding_size)

    def forward(self, x):
        # expect x: (B, C, T, V)
        x = self.layers(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)  # (B, channels)
        emb = self.embedding(x)
        emb = self.bn_emb(emb)
        return emb


# -----------------------------
# ArcFace Head
# -----------------------------

class ArcFace(nn.Module):
    """
    ArcFace layer: computes angular margin softmax. Returns logits ready for CrossEntropy.
    Implementation follows Additive Angular Margin Loss (ArcFace).
    """
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
        # emb: (B, D) not necessarily normalized; we will L2-normalize
        # labels: (B,) int
        emb_norm = F.normalize(emb, p=2, dim=1)
        W = F.normalize(self.weight, p=2, dim=1)  # (num_classes, D)
        logits = torch.matmul(emb_norm, W.t())  # (B, num_classes), cosine similarities
        if labels is None:
            # inference: return scaled logits
            return logits * self.s
        # training: apply additive angular margin
        cosine = logits
        sine = torch.sqrt(torch.clamp(1.0 - cosine ** 2, min=0.0))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)
        # if cosine > th: use phi, else: use cosine - mm
        phi_cond = cosine - self.mm
        # gather labels one-hot
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1,1), 1.0)
        # apply
        logits_with_margin = torch.where(cosine > self.th, phi, phi_cond)
        output = cosine * (1 - one_hot) + logits_with_margin * one_hot
        output = output * self.s
        return output


# -----------------------------
# Training utilities
# -----------------------------

@torch.no_grad()
def evaluate(model_backbone, classifier, dataloader, device):
    model_backbone.eval()
    classifier.eval()
    all_emb = []
    all_labels = []
    for x, y in dataloader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        emb = model_backbone(x)
        emb = F.normalize(emb, p=2, dim=1)
        all_emb.append(emb.cpu())
        all_labels.append(y.cpu())
    all_emb = torch.cat(all_emb, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    # simple evaluation: classification accuracy via classifier (logits)
    with torch.no_grad():
        logits = classifier(all_emb.to(device)) if isinstance(classifier, ArcFace) else classifier(all_emb.to(device), None)
        preds = logits.argmax(dim=1).cpu()
        acc = (preds == all_labels).float().mean().item()
    return acc


# -----------------------------
# Main training loop
# -----------------------------

def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    # dataset
    dataset = LandmarkVideoDataset(args.data_root, L=args.L, augmentation=True, compute_deltas=True)
    num_classes = len(dataset.id2idx)
    V = None
    # get V by loading first sample
    if len(dataset) > 0:
        sample, _ = dataset[0]
        # sample shape (C, L, V)
        V = sample.shape[2]
    else:
        raise RuntimeError('No data found')

    # pk sampler
    sampler = PKSampler(dataset, args.batch_p, args.batch_k)
    batch_size = args.batch_p * args.batch_k
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True)

    # create val loader that is deterministic (no augmentation)
    val_dataset = LandmarkVideoDataset(args.data_root, L=args.L, augmentation=False, compute_deltas=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # model
    in_channels = next(iter(dataloader))[0].shape[1]  # C
    backbone = STGCNBackbone(in_channels=in_channels, num_layers=9, V=V, base_channels=64, embedding_size=args.embedding_size, dropout=args.dropout)
    if args.use_arcface:
        classifier = ArcFace(args.embedding_size, num_classes, s=args.s, m=args.m)
    else:
        # simple linear classifier
        classifier = nn.Linear(args.embedding_size, num_classes)

    backbone.to(device)
    classifier.to(device)

    # optimizer
    opt = AdamW(list(backbone.parameters()) + list(classifier.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.num_epochs)

    # optionally resume
    start_epoch = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        backbone.load_state_dict(ck['backbone'])
        classifier.load_state_dict(ck['classifier'])
        opt.load_state_dict(ck['opt'])
        start_epoch = ck.get('epoch', 0) + 1
        print(f'Resumed from epoch {start_epoch}')

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # training loop
    best_val_acc = 0.0
    for epoch in range(start_epoch, args.num_epochs):
        backbone.train()
        classifier.train()
        running_loss = 0.0
        correct = 0
        total = 0
        t0 = time.time()
        last_preds = None
        last_labels = None
        last_correct = None

        for i, (x, labels) in enumerate(dataloader):
            x = x.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=args.amp):
                emb = backbone(x)  # (B, D)
                if args.use_arcface:
                    logits = classifier(emb, labels)
                else:
                    logits = classifier(emb)
                loss = F.cross_entropy(logits, labels)

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += x.size(0)

            # Save last batch info
            last_preds = preds.cpu().numpy()
            last_labels = labels.cpu().numpy()
            last_correct = (preds == labels).sum().item(), x.size(0)

        # Print only for the last batch
        print("Last batch preds:", last_preds)
        print("Last batch labels:", last_labels)
        print("Last batch correct:", f"{last_correct[0]} / {last_correct[1]}")

        scheduler.step()
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        t1 = time.time()

        # validation
        val_acc = evaluate(backbone, classifier, val_loader, device)

        print(f'Epoch {epoch+1}/{args.num_epochs} - loss {epoch_loss:.4f} acc {epoch_acc:.4f} val_acc {val_acc:.4f} time {(t1-t0):.1f}s')

        # checkpoint
        ckpt = {
            'backbone': backbone.state_dict(),
            'classifier': classifier.state_dict(),
            'opt': opt.state_dict(),
            'epoch': epoch
        }

        if (epoch + 1) == 1 or (epoch + 1) % 10 == 0:
            torch.save(ckpt, os.path.join(args.save_dir, f'ckpt_epoch_{epoch+1:03d}.pt'))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(ckpt, os.path.join(args.save_dir, 'best.pt'))

    print('Training complete')


if __name__ == '__main__':
    main()