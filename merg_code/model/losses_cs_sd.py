
import torch.nn.functional as F

def loss_ccl(C_s, C_v, C_s_gold, C_v_gold):
    return F.mse_loss(C_s, C_s_gold) + F.mse_loss(C_v, C_v_gold)

def loss_sal(S_s, S_v, S_s_gold, S_v_gold):
    return F.mse_loss(S_s, S_s_gold) + F.mse_loss(S_v, S_v_gold)

def loss_cls(logits, labels):
    ce = F.cross_entropy
    return (ce(logits['emotion'], labels['emotion'])
          + ce(logits['age'],     labels['age'])
          + ce(logits['gender'],  labels['gender'])
          + ce(logits['tone'],    labels['tone']))
