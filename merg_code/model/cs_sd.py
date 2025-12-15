import torch
import torch.nn as nn
import torch.nn.functional as F


class Reparam(nn.Module):
    def forward(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

def make_transformer(d_model=768, nhead=8, num_layers=4, dim_ff=2048, dropout=0.1, encoder=True):
    """✅ 근본 수정: Decoder는 norm_first=False"""
    layer = nn.TransformerEncoderLayer if encoder else nn.TransformerDecoderLayer
    mod = nn.TransformerEncoder if encoder else nn.TransformerDecoder

    # Encoder: norm_first=True, Decoder: norm_first=False
    norm_first = encoder  # ✅ 핵심 수정!
    l = layer(d_model, nhead, dim_ff, dropout, batch_first=True, norm_first=norm_first)
    return mod(l, num_layers=num_layers)

# latent-level fusion
def fuse_gaussian_poe(mu_s, logvar_s, mu_v, logvar_v, eps=1e-8):
    # σ² = exp(logvar)
    var_s = torch.exp(logvar_s)
    var_v = torch.exp(logvar_v)

    precision_s = 1.0 / (var_s + eps)
    precision_v = 1.0 / (var_v + eps)

    precision_joint = precision_s + precision_v
    var_joint = 1.0 / (precision_joint + eps)

    mu_joint = (mu_s * precision_s + mu_v * precision_v) * var_joint

    logvar_joint = torch.log(var_joint + eps)
    return mu_joint, logvar_joint


class ContentSynchronizer(nn.Module):
    """논문 4.3: Transformer-based VAE, z_c^{s/v} = EncCS(FFN(r_t), q_c^{s/v})"""

    def __init__(self, d_in=4096, d_latent=512, d_out=768, num_layers=4, nhead=8, dim_ff=2048, qdim=768):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.ffn_in = nn.Linear(d_in, d_out)
        self.enc = make_transformer(d_model=d_out, nhead=nhead, num_layers=num_layers, dim_ff=dim_ff, encoder=True)

        self.to_mu = nn.Linear(d_out, d_latent)
        self.to_logvar = nn.Linear(d_out, d_latent)
        self.reparam = Reparam()
        self.latent_to_mem = nn.Linear(d_latent, d_out)

        self.q_s_c = nn.Parameter(torch.randn(1, 1, qdim))
        self.q_v_c = nn.Parameter(torch.randn(1, 1, qdim))

        self.dec = make_transformer(d_model=d_out, nhead=nhead, num_layers=num_layers, dim_ff=dim_ff, encoder=False)
        self.proj_s = nn.Linear(d_out, d_out)
        self.proj_v = nn.Linear(d_out, d_out)

    def _decode(self, mem, q):
        """✅ 안정적 decoder: explicit mask + proper shape"""
        B = mem.size(0)
        device = mem.device
        dtype = mem.dtype

        mem = mem.to(dtype)  # [B, 1, 768]
        q = q.to(dtype)

        qB = q.expand(B, -1, -1)  # [B, 1, 768]

        # seq_len=1이므로 empty mask
        tgt_mask = torch.zeros((1, 1), dtype=torch.bool, device=device)

        with torch.no_grad():  # mask는 gradient 불필요
            out = self.dec(tgt=qB, memory=mem, tgt_mask=tgt_mask)
        return out[:, 0, :]  # [B, 768]

    def _cast_layers_to_input_dtype(self, dtype):
        for module in [self.ffn_in, self.enc, self.to_mu, self.to_logvar,
                       self.latent_to_mem, self.dec, self.proj_s, self.proj_v]:
            module.to(dtype)
        self.q_s_c.data = self.q_s_c.data.to(dtype)
        self.q_v_c.data = self.q_v_c.data.to(dtype)

    def forward(self, r_t, return_kld=True):
        """✅ Dynamic shape: [B, T, 4096] or [B*T, 4096]"""
        self._cast_layers_to_input_dtype(r_t.dtype) # dtype: float16

        orig_shape = r_t.shape
        was_flat = len(r_t.shape) == 2

        # ✅ Input reshape
        if was_flat:  # [B*T, 4096]
            B_T = r_t.size(0)
            r_t = r_t.unsqueeze(1)  # [B*T, 1, 4096]

        # 1. FFN -> Encoder
        x = self.ffn_in(r_t)  # [B, T, 768]
        h_enc = self.enc(x)  # [B, T, 768]
        pooled = h_enc.mean(dim=1)  # [B, 768]

        # 2. VAE
        mu = self.to_mu(pooled)
        logvar = self.to_logvar(pooled)
        z_c = self.reparam(mu, logvar)
        mem = self.latent_to_mem(z_c).unsqueeze(1)  # [B, 1, 768]

        # 3. Decode
        C_s_raw = self._decode(mem, self.q_s_c)
        C_v_raw = self._decode(mem, self.q_v_c)
        C_s = self.proj_s(C_s_raw)
        C_v = self.proj_v(C_v_raw)

        # 4. KLD
        kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean() if return_kld else 0.0

        # ✅ Shape 복원
        if was_flat:
            C_s = C_s.view(orig_shape[0], -1).mean(dim=1)
            C_v = C_v.view(orig_shape[0], -1).mean(dim=1)

        return C_s, C_v, kld


class StyleDisentangler(nn.Module):
    """논문 4.3: disentangle E_s/v, P_s/v from r_s, r_v -> S_s/v = E_s/v ⊕ P_s/v"""

    def __init__(self, d_in=4096, d_latent=256, d_out=768, num_layers=4, nhead=8, dim_ff=2048, qdim=768,
                 n_emotions=7, n_age=4, n_gender=2, n_tone=3):
        super().__init__()

        self.ffn_s = nn.Linear(d_in, d_out)
        self.ffn_v = nn.Linear(d_in, d_out)
        self.enc_s = make_transformer(d_model=d_out, nhead=nhead, num_layers=num_layers, dim_ff=dim_ff, encoder=True)
        self.enc_v = make_transformer(d_model=d_out, nhead=nhead, num_layers=num_layers, dim_ff=dim_ff, encoder=True)

        self.to_mu_e = nn.Linear(d_out, d_latent)
        self.to_logvar_e = nn.Linear(d_out, d_latent)
        self.to_mu_p = nn.Linear(d_out, d_latent)
        self.to_logvar_p = nn.Linear(d_out, d_latent)
        self.reparam = Reparam()

        self.q_s_e = nn.Parameter(torch.randn(1, 1, qdim))
        self.q_v_e = nn.Parameter(torch.randn(1, 1, qdim))
        self.q_s_p = nn.Parameter(torch.randn(1, 1, qdim))
        self.q_v_p = nn.Parameter(torch.randn(1, 1, qdim))

        self.latent_to_mem_e = nn.Linear(d_latent, d_out)
        self.latent_to_mem_p = nn.Linear(d_latent, d_out)
        self.dec = make_transformer(d_model=d_out, nhead=nhead, num_layers=num_layers, dim_ff=dim_ff, encoder=False)

        self.head_e = nn.Linear(d_out, d_out)
        self.head_p = nn.Linear(d_out, d_out)
        self.fuser_s = nn.Linear(d_out * 2, 192)
        self.fuser_v = nn.Linear(d_out * 2, d_out) # StyleTTS2->JDCNet output dim

        self.cls_emotion = nn.Linear(d_out, n_emotions)
        self.cls_age = nn.Linear(d_out, n_age)
        self.cls_gender = nn.Linear(d_out, n_gender)
        self.cls_tone = nn.Linear(d_out, n_tone)

    def _encode_pool(self, enc, ffn, x):
        h = enc(ffn(x))
        return h.mean(dim=1)

    def _decode_query(self, mem, q, proj):
        """z -> E/P using query"""
        B = mem.size(0)
        qB = q.expand(B, -1, -1)
        out = self.dec(tgt=qB, memory=mem)
        return proj(out[:, 0, :])  # [B, 768] # aisha에 있던 mask 코드 삭제(유진)

    def _cast_layers_to_input_dtype(self, dtype):
        for module in [self.ffn_s, self.ffn_v, self.enc_s, self.enc_v, self.to_mu_e, self.to_mu_p, self.to_logvar_e, self.to_logvar_p,
                       self.q_s_e, self.q_v_e, self.q_s_p, self.q_v_p, self.latent_to_mem_e, self.latent_to_mem_p,
                       self.dec, self.head_e, self.head_p, self.fuser_s, self.fuser_v,
                       self.cls_emotion, self.cls_age, self.cls_gender, self.cls_tone]:
            module.to(dtype)

    def _cast_layers_to_input_dtype(self, dtype):
        # Linear / Transformer 모듈들 dtype 통일
        for module in [
            self.ffn_s, self.ffn_v, self.enc_s, self.enc_v,
            self.to_mu_e, self.to_logvar_e, self.to_mu_p, self.to_logvar_p,
            self.latent_to_mem_e, self.latent_to_mem_p,
            self.dec, self.head_e, self.head_p, self.fuser_s, self.fuser_v,
            self.cls_emotion, self.cls_age, self.cls_gender, self.cls_tone,]:
            module.to(dtype)

        # learnable query 파라미터들 dtype 통일
        self.q_s_e.data = self.q_s_e.data.to(dtype)
        self.q_v_e.data = self.q_v_e.data.to(dtype)
        self.q_s_p.data = self.q_s_p.data.to(dtype)
        self.q_v_p.data = self.q_v_p.data.to(dtype)

    def forward(self, r_s, r_v, return_kld=True):
        """✅ Dynamic shape 지원"""
        self._cast_layers_to_input_dtype(r_s.dtype)

        def reshape_input(x):
            orig_shape = x.shape
            was_flat = len(x.shape) == 2
            if was_flat:
                x = x.unsqueeze(1)
            return x, orig_shape, was_flat

        r_s, orig_s, flat_s = reshape_input(r_s)
        r_v, orig_v, flat_v = reshape_input(r_v)

        # 1. Encode
        hs = self._encode_pool(self.enc_s, self.ffn_s, r_s)
        hv = self._encode_pool(self.enc_v, self.ffn_v, r_v)

        ####################수정부분###################
        mu_e_s, logvar_e_s = self.to_mu_e(hs), self.to_logvar_e(hs)
        mu_e_v, logvar_e_v = self.to_mu_e(hv), self.to_logvar_e(hv)

        mu_e, logvar_e = fuse_gaussian_poe(mu_e_s, logvar_e_s, mu_e_v, logvar_e_v)
        temp = 1.5
        logvar_e_scaled = logvar_e + 2 * torch.log(torch.tensor(temp, device=logvar_e.device, dtype=logvar_e.dtype))
        z_fusion = self.reparam(mu_e, logvar_e_scaled)
        ####################수정부분###################

        # 2. Emotion disentangling (논문 수식 3-4)
        #mu_e_s, logvar_e_s = self.to_mu_e(hs), self.to_logvar_e(hs)
        #mu_e_v, logvar_e_v = self.to_mu_e(hv), self.to_logvar_e(hv)
        #z_e_s = self.reparam(mu_e_s, logvar_e_s)
        #z_e_v = self.reparam(mu_e_v, logvar_e_v)
        mem_e_s = self.latent_to_mem_e(z_fusion).unsqueeze(1)
        mem_e_v = self.latent_to_mem_e(z_fusion).unsqueeze(1)
        E_s = self._decode_query(mem_e_s, self.q_s_e, self.head_e)
        E_v = self._decode_query(mem_e_v, self.q_v_e, self.head_e)

        # 3. Profile disentangling (논문 수식 5-6)
        mu_p_s, logvar_p_s = self.to_mu_p(hs), self.to_logvar_p(hs)
        mu_p_v, logvar_p_v = self.to_mu_p(hv), self.to_logvar_p(hv)
        z_p_s = self.reparam(mu_p_s, logvar_p_s)
        z_p_v = self.reparam(mu_p_v, logvar_p_v)
        mem_p_s = self.latent_to_mem_p(z_p_s).unsqueeze(1)
        mem_p_v = self.latent_to_mem_p(z_p_v).unsqueeze(1)
        P_s = self._decode_query(mem_p_s, self.q_s_p, self.head_p)
        P_v = self._decode_query(mem_p_v, self.q_v_p, self.head_p)

        # 4. Style fusion (논문 수식 7)
        S_s = self.fuser_s(torch.cat([E_s, P_v], dim=-1))
        S_v = self.fuser_v(torch.cat([E_v, P_v], dim=-1))
        #S_s = self.fuser_s(torch.cat([E_s, P_s], dim=-1))  # [B, 768]
        #S_v = self.fuser_v(torch.cat([E_v, P_v], dim=-1))

        # 5. Global features for classification supervision (논문 D.3 Step3)
        E_global = 0.5 * (E_s + E_v)  # fuse E_s, E_v
        P_global = 0.5 * (P_s + P_v)  # fuse P_s, P_v
        logits = {
            'emotion': self.cls_emotion(E_global),
            'age': self.cls_age(P_global),
            'gender': self.cls_gender(P_global),
            'tone': self.cls_tone(P_global) # tone = timbre
        }

        # 6. KLD loss (training에서만 사용)
        kld = 0.0
        if return_kld:
            for mu, logvar in [(mu_e_s, logvar_e_s), (mu_e_v, logvar_e_v),
                               (mu_p_s, logvar_p_s), (mu_p_v, logvar_p_v)]:
                kld += -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()

        return S_s, S_v, logits, kld


# 테스트
if __name__ == "__main__":
    C_s = torch.load('/home/ahlee/bk/multimodal-Empatheia/merg_code/model/cs_sd_tensor/C_s.pt') # (1,768)
    C_v = torch.load('/home/ahlee/bk/multimodal-Empatheia/merg_code/model/cs_sd_tensor/C_v.pt') # (1,768)
    S_s = torch.load('/home/ahlee/bk/multimodal-Empatheia/merg_code/model/cs_sd_tensor/S_s.pt') # (1,768)
    S_v = torch.load('/home/ahlee/bk/multimodal-Empatheia/merg_code/model/cs_sd_tensor/S_v.pt') # (1,768)

    # Test 1: Normal shape
    B, T = 4, 10
    r_t = torch.randn(B, T, 4096)
    r_s = torch.randn(B, T, 4096)
    r_v = torch.randn(B, T, 4096)

    cs = ContentSynchronizer(d_in=4096, d_latent=512, d_out=768,
                             num_layers=4, nhead=8, dim_ff=2048).to('cuda')
    sd = StyleDisentangler(d_in=4096, d_latent=256, d_out=768,
                           num_layers=4, nhead=8, dim_ff=2048).to('cuda')
    C_s, C_v, kld_cs = cs(r_t.to('cuda'))
    S_s, S_v, logits, kld_sd = sd(r_s.to('cuda'), r_v.to('cuda'))