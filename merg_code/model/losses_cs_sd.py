import torch
import torch.nn.functional as F


def loss_ccl(C_s, C_v, C_s_gold, C_v_gold):
    return F.mse_loss(C_s, C_s_gold) + F.mse_loss(C_v, C_v_gold)


def loss_sal(S_s, S_v, S_s_gold, S_v_gold):
    return F.mse_loss(S_s, S_s_gold) + F.mse_loss(S_v, S_v_gold)


def _safe_ce_cpu(name, logits, labels):
    """
    Compute cross entropy on CPU, with labels clamped to [0, C-1].
    This completely avoids CUDA NLL asserts even if labels are weird.
    """
    # logits: [B, C] on *GPU*
    # labels: [B] on CPU or GPU or plain list

    # Make sure labels are a 1D LongTensor
    if not torch.is_tensor(labels):
        labels = torch.tensor(labels, dtype=torch.long)
    else:
        labels = labels.to(dtype=torch.long)

    # Number of classes
    n_classes = logits.size(-1)

    # Clamp labels to valid range
    labels = labels.clamp(0, n_classes - 1)

    # Move to CPU for loss computation
    logits_cpu = logits.detach().cpu()
    labels_cpu = labels.cpu()

    # --- tiny debug: print ranges once in a while ---
    try:
        lmin = int(labels_cpu.min().item())
        lmax = int(labels_cpu.max().item())
    except ValueError:
        # empty tensor case (should not happen, but safe-guard)
        lmin, lmax = -1, -1

    # Only print occasionally to avoid spam
    if torch.rand(1).item() < 0.001:
        print(f"[loss_cls:{name}] n_classes={n_classes}, labels_min={lmin}, labels_max={lmax}")

    loss = F.cross_entropy(logits_cpu, labels_cpu)
    # Move scalar back to the same device as logits
    return loss.to(logits.device)


def loss_cls(logits, labels):
    """
    logits: dict with keys 'emotion', 'age', 'gender', 'tone'
            each [B, C_k]
    labels: dict with same keys, each [B] (Long or list[int])
    We compute each head's CE on CPU via _safe_ce_cpu to dodge CUDA asserts.
    """
    total = 0.0

    for key in ["emotion", "age", "gender", "tone"]:
        if key not in logits or key not in labels:
            continue

        # label이 list 형태라면 tensor로 변환
        target = labels[key]
        if isinstance(target, list):
            target = torch.tensor(target, dtype=torch.long, device=logits[key].device)
        else:
            target = target.to(dtype=torch.long, device=logits[key].device)

        total = total + F.cross_entropy(logits[key], target)

    return total
