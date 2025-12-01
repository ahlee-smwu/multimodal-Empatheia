
import torch, torch.nn as nn, os

class StyleTTS2Encoders(nn.Module):
    def __init__(self, ckpt_dir):
        super().__init__()
        self.text_aco = torch.jit.load(os.path.join(ckpt_dir, 'text_aco_encoder.pt'))
        self.text_bert= torch.jit.load(os.path.join(ckpt_dir, 'text_bert_encoder.pt'))
        self.ref_enc  = torch.jit.load(os.path.join(ckpt_dir, 'reference_encoder.pt'))
        for m in [self.text_aco, self.text_bert, self.ref_enc]:
            m.eval().requires_grad_(False)

    @torch.no_grad()
    def text_content(self, texts):
        em_aco = self.text_aco(texts)   # [B, D]
        em_bert= self.text_bert(texts)  # [B, D]
        # If encoders output 768 each, concat->1536. Reduce to 768 so MSE dims match.
        if em_aco.shape[-1] == em_bert.shape[-1]:
            import torch.nn.functional as F
            cat = torch.cat([em_aco, em_bert], dim=-1)
            # project to 768 with a fixed random ortho matrix stored as buffer? Keep simple: linear on the fly.
            W = torch.zeros(cat.shape[-1], em_aco.shape[-1], device=cat.device, dtype=cat.dtype)
            torch.nn.init.xavier_uniform_(W)
            return cat @ W
        return torch.cat([em_aco, em_bert], dim=-1)

    @torch.no_grad()
    def style_from_audio(self, wav):
        return self.ref_enc(wav)
