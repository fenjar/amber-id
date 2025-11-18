import torch
import torch.nn.functional as F

# def nt_xent_loss(embeddings, labels, temperature=0.1):
#     embeddings = F.normalize(embeddings, dim=1)
#     device = embeddings.device
#     sim_matrix = torch.matmul(embeddings, embeddings.T) / temperature
#     self_mask = torch.eye(sim_matrix.size(0), device=device).bool()
#     sim_matrix = sim_matrix.masked_fill(self_mask, float('-inf'))
#     labels = labels.to(device)
#     labels = labels.contiguous().view(-1, 1)
#     pos_mask = torch.eq(labels, labels.T).float()
#     pos_mask = pos_mask.masked_fill(self_mask, 0)
#     if hasattr(pos_mask, "fill_diagonal_"):
#         pos_mask.fill_diagonal_(0)
#     else:
#         pos_mask = pos_mask - torch.diag(torch.diag(pos_mask))
#     log_prob = F.log_softmax(sim_matrix, dim=1)
#     # Mask diagonal in log_prob as well
#     if hasattr(log_prob, "fill_diagonal_"):
#         log_prob.fill_diagonal_(0)
#     else:
#         log_prob = log_prob - torch.diag(torch.diag(log_prob))
#     valid = pos_mask.sum(1) > 0
#     if valid.sum() == 0:
#         return torch.tensor(0.0, device=device, requires_grad=True)
#     numerator = (pos_mask * log_prob).sum(1)
#     denominator = pos_mask.sum(1) + 1e-8
#     mean_log_prob_pos = numerator[valid] / denominator[valid]
#     loss = -mean_log_prob_pos.mean()
#     print("pos_mask diagonal:", pos_mask.diag())
#     print("log_prob diagonal:", log_prob.diag())
#     print("[LOGGING] numerator has nan:", torch.isnan(numerator).any().item())
#     print("[LOGGING] loss is nan:", torch.isnan(loss).item())
#     return loss

def nt_xent_loss(embeddings, labels, temperature=0.1):
    embeddings = F.normalize(embeddings, dim=1)
    device = embeddings.device
    sim_matrix = torch.matmul(embeddings, embeddings.T) / temperature
    self_mask = torch.eye(sim_matrix.size(0), device=device).bool()
    sim_matrix = sim_matrix.masked_fill(self_mask, float('-inf'))
    labels = labels.to(device)
    labels = labels.contiguous().view(-1, 1)
    pos_mask = torch.eq(labels, labels.T).float()
    pos_mask = pos_mask.masked_fill(self_mask, 0)
    log_prob = F.log_softmax(sim_matrix, dim=1)
    log_prob = log_prob.masked_fill(self_mask, 0)  # <--- out-of-place, safe for autograd
    valid = pos_mask.sum(1) > 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    numerator = (pos_mask * log_prob).sum(1)
    denominator = pos_mask.sum(1) + 1e-8
    mean_log_prob_pos = numerator[valid] / denominator[valid]
    loss = -mean_log_prob_pos.mean()
    return loss

def exp_sim(a, b, tau=0.2):
    """
    Similarity metric s(a,b) = exp(-||a-b||^2 / tau)
    a: (N, D)
    b: (M, D)
    """
    #print(f"a.shape: {a.shape}, b.shape: {b.shape}")
    # a = torch.nn.functional.normalize(a, dim=1)
    # b = torch.nn.functional.normalize(b, dim=1)
    # dist_sq = torch.cdist(a, b, p=2).pow(2)  # (N, M)
    # return torch.exp(-dist_sq / tau)
    a = torch.nn.functional.normalize(a, dim=1)
    b = torch.nn.functional.normalize(b, dim=1)
    return torch.mm(a, b.t())

def compute_N_term(embeddings, driver_ids, video_ids, tau=0.2):
    """
    N-Term: Sum of max similarities to all videos of the same driver.
    Normalized by number of videos to avoid favoring drivers with many videos.
    """
    device = embeddings.device
    N_vals = torch.zeros(len(embeddings), device=device, dtype=embeddings.dtype)

    for driver in torch.unique(driver_ids):
        mask_ref = driver_ids == driver
        ref_emb = embeddings[mask_ref]  # all clips of this driver
        ref_idx = torch.where(mask_ref)[0]

        # all unique videos of this driver
        vids = torch.unique(video_ids[mask_ref])

        for idx_ref in ref_idx:
            emb_ref = embeddings[idx_ref:idx_ref+1]  # (1, D)
            total_sim = 0.0
            for vid in vids:
                mask_vid = (video_ids == vid) & (driver_ids == driver)
                emb_vid = embeddings[mask_vid]
                if emb_vid.size(0) == 0:
                    continue
                sims = exp_sim(emb_ref, emb_vid, tau=tau).squeeze(0)  # (num_clips_vid,)
                total_sim += sims.max()
            N_vals[idx_ref] = total_sim / len(vids)  # Normalisierung
    return N_vals

def compute_Q_term(embeddings, driver_ids, tau=0.2):
    """
    Q-Term: Max similarity to all clips of other drivers.
    """
    device = embeddings.device
    Q_vals = torch.zeros(len(embeddings), device=device, dtype=embeddings.dtype)

    for driver in torch.unique(driver_ids):
        mask_ref = driver_ids == driver
        ref_emb = embeddings[mask_ref]
        ref_idx = torch.where(mask_ref)[0]

        neg_emb = embeddings[driver_ids != driver]
        if neg_emb.size(0) == 0:
            continue

        sims_all = exp_sim(ref_emb, neg_emb, tau=tau)  # (num_ref, num_neg)
        max_vals = sims_all.max(dim=1)[0]
        #print(f"[DEBUG] max_vals.shape: {max_vals.shape}, Q_vals: {Q_vals.shape}, ref_idx: {ref_idx.shape}")
        #print(f"[DEBUG] max_vals: {max_vals}, Q_vals: {Q_vals}, ref_idx: {ref_idx}")
        # Q_vals[ref_idx] = max_vals
        if ref_idx.numel() == max_vals.numel():
            Q_vals[ref_idx] = max_vals
        else:
            for i, idx in enumerate(ref_idx):
                Q_vals[idx] = max_vals[i]
    return Q_vals

def compute_R_term(embeddings, clip_is_shuffled, original_pair_idx, tau=0.2):
    """
    R-Term: Similarity between original clip and its shuffled counterpart.
    - clip_is_shuffled: Bool tensor indicating which clips are shuffled
    - original_pair_idx: Mapping from shuffled clip to index of its original
    """
    device = embeddings.device
    R_vals = torch.zeros(len(embeddings), device=device, dtype=embeddings.dtype)

    shuffled_idx = torch.where(clip_is_shuffled)[0]
    for idx in shuffled_idx:
        orig_idx = original_pair_idx[idx]
        sim_val = exp_sim(
            embeddings[idx:idx+1], embeddings[orig_idx:orig_idx+1], tau=tau
        ).item()
        R_vals[idx] = sim_val
        R_vals[orig_idx] = sim_val  # symmetrisch
    return R_vals

def avatar_contrastive_loss(embeddings, driver_ids, video_ids,
                            clip_is_shuffled, original_pair_idx,
                            tau=0.1, wN=1.0, wQ=1.0, wR=0.5):
    
    """
    Computes the combined loss as in the NVIDIA paper, adapted for mapping on one avatar.
    """
    dist_matrix = torch.cdist(embeddings, embeddings, p=2)
    mean_dist = dist_matrix.mean().detach()
    tau_adaptive = tau * (1 + 0.5 * mean_dist.item())

    N_vals = compute_N_term(embeddings, driver_ids, video_ids, tau=tau_adaptive)
    Q_vals = compute_Q_term(embeddings, driver_ids, tau=tau_adaptive)
    R_vals = compute_R_term(embeddings, clip_is_shuffled, original_pair_idx, tau=tau_adaptive)

    denom = wN * N_vals + wQ * Q_vals + wR * R_vals + 1e-8
    p_vals = (wN * N_vals) / denom
    loss = -torch.log(p_vals + 1e-8).mean()

    return loss, {"N": N_vals.mean().item(),
                  "Q": Q_vals.mean().item(),
                  "R": R_vals.mean().item(),
                  "p": p_vals.mean().item()}