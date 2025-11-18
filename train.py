import os
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from sampler import AvatarClipDataset, build_driver_to_videos, avatar_collate_fn, expand_pair_idx_by_5, expand_by_5, preprocess_dataset
from model import TemporalIDNet3D
from loss import avatar_contrastive_loss, nt_xent_loss
from landmarks import get_global_landmark_indices

def test_gpu():
    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print()

    #Additional Info when using cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
        print("Torch version:", torch.__version__)
        print("CUDA available:", torch.cuda.is_available())
        print("CUDA device count:", torch.cuda.device_count())
        print("Current device:", torch.cuda.current_device())
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        device = torch.device('cuda:1')
        total = torch.cuda.get_device_properties(device).total_memory
        reserved = torch.cuda.memory_reserved(device)
        allocated = torch.cuda.memory_allocated(device)
        free = reserved - allocated

        print(f"Total GPU memory: {total / 1024**3:.2f} GB")
        print(f"Reserved memory: {reserved / 1024**3:.2f} GB")
        print(f"Allocated memory: {allocated / 1024**3:.2f} GB")
        print(f"Free (unallocated in reserved): {free / 1024**3:.2f} GB")
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

# def setup_distributed():
#     """Initialization of Distributed Training via Slurm-Environment Variables."""
#     dist.init_process_group("nccl")
#     local_rank = int(os.environ.get("LOCAL_RANK", 0))
#     torch.cuda.set_device(local_rank)
#     return local_rank

def setup_distributed():
    dist.init_process_group("nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    print(f"LOCAL_RANK from env: {local_rank}")
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices is not None and "," not in cuda_visible_devices:
        # Only one GPU is visible to this process
        torch.cuda.set_device(0)
    else:
        # All GPUs are visible, use local_rank
        torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_distributed():
    dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, help="Directory to the avatars dataset")
    parser.add_argument("--work_dir", type=str, required=True, help="Directory to save checkpoints and logs")
    parser.add_argument("--F", type=int, default=71, help="Clip length")
    parser.add_argument("--batch_identities", type=int, default=8, help="Batch size per identity")
    parser.add_argument("--clips_per_id", type=int, default=16, help="Number of clips per identity")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--amp", action="store_true", help="Use Automatic Mixed Precision")
    parser.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--num_landmarks", type=int, default=126, help="Number of landmarks per frame")
    parser.add_argument("--preprocessing_only", action="store_true", help="Only preprocess dataset and exit")
    parser.add_argument("--preprocessed_dir", type=str, default=None, help="Directory for preprocessed clips (if set to None the preprocessing will be donne on-the-fly during training)")
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()

    # === Distributed Setup ===
    local_rank = setup_distributed()
    print(f"[rank{local_rank}] CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    device = torch.device("cuda", local_rank)
    is_master = (dist.get_rank() == 0)

    selected_indices = get_global_landmark_indices(args.num_landmarks)

    if args.preprocessing_only:
        driver_to_videos = build_driver_to_videos(args.data_root)
        preprocess_dataset(driver_to_videos, args.work_dir, args.clips_per_id, args.num_landmarks, args.F+4, selected_indices)
        print("Preprocessing finished.")
        return

    # === Dataset / Sampler / Loader ===
    #dataset = AvatarClipDataset(args.data_root, F=args.F, clips_per_id=args.clips_per_id)
    driver_to_videos = build_driver_to_videos(args.data_root, preprocessed=(args.preprocessed_dir is not None))
    dataset = AvatarClipDataset(driver_to_videos, clips_per_id=args.clips_per_id, num_landmarks=args.num_landmarks, clip_length=args.F, preprocessed_dir=args.preprocessed_dir, selected_indices=selected_indices)
    sampler = DistributedSampler(dataset, shuffle=True)
    loader = DataLoader(dataset, batch_size=args.batch_identities, sampler=sampler,
                        num_workers=4, pin_memory=True, collate_fn=avatar_collate_fn)

    # === Model ===
    # Calculate D based on num_landmarks
    D = (args.num_landmarks * (args.num_landmarks - 1)) // 2

    model = TemporalIDNet3D(F=args.F,  D=D, embed_dim=128).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # === Optimizer ===
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    start_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
            checkpoint = torch.load(args.resume, map_location=map_location)
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scaler_state_dict' in checkpoint and args.amp:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
                print(f"Resumed from checkpoint {args.resume}, starting at epoch {start_epoch}")
    
    # === Training Loop ===
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)  # important for DDP-Shuffling
        model.train()
        print(f"Number of batches per epoch: {len(loader)}")
        batch_iter = tqdm(loader, desc=f"Epoch {epoch+1}") if is_master else loader
        for step, batch in enumerate(batch_iter):
            # batch = (segments, driver_ids, video_ids)
            # print(f"[TRAIN] batch type: {type(batch)}, batch len: {len(batch) if hasattr(batch, '__len__') else 'N/A'}")
            # print(f"[TRAIN] batch shape: {[b.shape if isinstance(b, torch.Tensor) else str(type(b)) for b in batch]}")
            segments, driver_ids, video_ids, clip_is_shuffled, original_pair_idx = batch
            segments = segments.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=args.amp):
                # print(f"[TRAIN] segments.shape: {segments.shape}, F: {args.F}")
                embeddings = model(segments)   # (B, 5, E)
                embeddings = embeddings.reshape(-1, embeddings.size(-1))  # [B*5, 128]
                driver_ids = expand_by_5(driver_ids)
                video_ids = expand_by_5(video_ids)
                clip_is_shuffled = expand_by_5(clip_is_shuffled)
                original_pair_idx = expand_pair_idx_by_5(original_pair_idx)

                loss, loss_info = avatar_contrastive_loss(
                    embeddings, driver_ids, video_ids, clip_is_shuffled, original_pair_idx
                    )

                # --- Use NT-Xent/InfoNCE loss ---
                # loss = nt_xent_loss(embeddings, driver_ids, temperature=0.1)
                # loss_info = {}  # Optionally, you can log additional info if needed                

                
                if is_master and step % 10 == 0:
                    print("Embeddings mean:", embeddings.mean().item(), "std:", embeddings.std().item())
                    # After embeddings = model(segments) and embeddings = embeddings.reshape(-1, embeddings.size(-1))
                    with torch.no_grad():
                        # Normalize embeddings for cosine similarity
                        normed_emb = torch.nn.functional.normalize(embeddings, dim=1)
                        # Compute cosine similarity matrix
                        sim_matrix = torch.mm(normed_emb, normed_emb.t())
                        # Exclude diagonal (self-similarity)
                        mask = ~torch.eye(sim_matrix.size(0), dtype=bool, device=sim_matrix.device)
                        sims = sim_matrix[mask]
                        print(f"[SIM] Cosine similarity stats: mean={sims.mean().item():.4f}, std={sims.std().item():.4f}, min={sims.min().item():.4f}, max={sims.max().item():.4f}")

            scaler.scale(loss).backward()

            # --- Compute gradient norm after backward ---
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            grad_norm = total_norm ** 0.5

            if (step + 1) % args.grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            if is_master and step % 10 == 0:
                print("Unique driver_ids:", torch.unique(driver_ids))
                print("Unique video_ids:", torch.unique(video_ids))
                print("Zero-padded clips:", (segments == 0).all(dim=(1,2)).sum().item())
                print(f"[TRAIN] Epoch {epoch} Step {step} Loss {loss:.4f} GradNorm {grad_norm:.4f}")
                print(f"Loss Info: " + ", ".join([f"{k}: {v:.4f}" for k, v in loss_info.items()]))

        if is_master and (epoch + 1) % 5 == 0:
            ckpt_path = os.path.join(args.work_dir, f"epoch_{epoch+1}.pt")
            # torch.save(model.module.state_dict(), ckpt_path)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
            }, ckpt_path)

    cleanup_distributed()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test_gpu":
        test_gpu()
    else:
        main()