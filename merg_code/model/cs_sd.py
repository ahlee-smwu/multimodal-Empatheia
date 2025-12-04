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
        print('mem', mem.shape)
        print('q', q.shape)
        B = mem.size(0)
        device = mem.device
        dtype = mem.dtype

        qB = q.expand(B, -1, -1)  # [B, 1, 768]
        mem = mem.to(dtype)  # [B, 1, 768]

        # seq_len=1이므로 empty mask
        tgt_mask = torch.zeros((1, 1), dtype=torch.bool, device=device)

        with torch.no_grad():  # mask는 gradient 불필요
            out = self.dec(tgt=qB, memory=mem, tgt_mask=tgt_mask)
        return out[:, 0, :]  # [B, 768]

    def _cast_layers_to_input_dtype(self, dtype):
        for module in [self.ffn_in, self.enc, self.to_mu, self.to_logvar,
                       self.latent_to_mem, self.proj_s, self.proj_v]:
            module.to(dtype)

    def forward(self, r_t, return_kld=True):
        """✅ Dynamic shape: [B, T, 4096] or [B*T, 4096]"""
        self._cast_layers_to_input_dtype(r_t.dtype)

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
        self.d_out = d_out

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
        self.fuser_s = nn.Linear(d_out * 2, d_out)
        self.fuser_v = nn.Linear(d_out * 2, d_out)

        self.cls_emotion = nn.Linear(d_out, n_emotions)
        self.cls_age = nn.Linear(d_out, n_age)
        self.cls_gender = nn.Linear(d_out, n_gender)
        self.cls_tone = nn.Linear(d_out, n_tone)

    def _encode_pool(self, enc, ffn, x):
        h = enc(ffn(x))
        return h.mean(dim=1)

    def _decode_query(self, mem, q, proj):
        """✅ StyleDisentangler용 decoder"""
        B = mem.size(0)
        device = mem.device
        dtype = mem.dtype

        qB = q.expand(B, -1, -1)
        mem = mem.to(dtype)

        tgt_mask = torch.zeros((1, 1), dtype=torch.bool, device=device)

        with torch.no_grad():
            out = self.dec(tgt=qB, memory=mem, tgt_mask=tgt_mask)
        return proj(out[:, 0, :])

    def forward(self, r_s, r_v, return_kld=True):
        """✅ Dynamic shape 지원"""

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

        # 2. Emotion
        mu_e_s, logvar_e_s = self.to_mu_e(hs), self.to_logvar_e(hs)
        mu_e_v, logvar_e_v = self.to_mu_e(hv), self.to_logvar_e(hv)
        z_e_s = self.reparam(mu_e_s, logvar_e_s)
        z_e_v = self.reparam(mu_e_v, logvar_e_v)
        mem_e_s = self.latent_to_mem_e(z_e_s).unsqueeze(1)
        mem_e_v = self.latent_to_mem_e(z_e_v).unsqueeze(1)
        E_s = self._decode_query(mem_e_s, self.q_s_e, self.head_e)
        E_v = self._decode_query(mem_e_v, self.q_v_e, self.head_e)

        # 3. Profile
        mu_p_s, logvar_p_s = self.to_mu_p(hs), self.to_logvar_p(hs)
        mu_p_v, logvar_p_v = self.to_mu_p(hv), self.to_logvar_p(hv)
        z_p_s = self.reparam(mu_p_s, logvar_p_s)
        z_p_v = self.reparam(mu_p_v, logvar_p_v)
        mem_p_s = self.latent_to_mem_p(z_p_s).unsqueeze(1)
        mem_p_v = self.latent_to_mem_p(z_p_v).unsqueeze(1)
        P_s = self._decode_query(mem_p_s, self.q_s_p, self.head_p)
        P_v = self._decode_query(mem_p_v, self.q_v_p, self.head_p)

        # 4. Style fusion
        S_s = self.fuser_s(torch.cat([E_s, P_s], dim=-1))
        S_v = self.fuser_v(torch.cat([E_v, P_v], dim=-1))

        # 5. Classification
        E_global = 0.5 * (E_s + E_v)
        P_global = 0.5 * (P_s + P_v)
        logits = {
            'emotion': self.cls_emotion(E_global),
            'age': self.cls_age(P_global),
            'gender': self.cls_gender(P_global),
            'tone': self.cls_tone(P_global)
        }

        # 6. KLD
        kld = 0.0
        if return_kld:
            for mu, logvar in [(mu_e_s, logvar_e_s), (mu_e_v, logvar_e_v),
                               (mu_p_s, logvar_p_s), (mu_p_v, logvar_p_v)]:
                kld += -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()

        # ✅ Shape 복원
        if flat_s:
            S_s = S_s.view(orig_s[0], -1).mean(dim=1)
        if flat_v:
            S_v = S_v.view(orig_v[0], -1).mean(dim=1)

        return S_s, S_v, logits, kld


# 테스트
if __name__ == "__main__":
    # Test 1: Normal shape
    B, T = 4, 10
    r_t = torch.randn(B, T, 4096)
    r_s = torch.randn(B, T, 4096)
    r_v = torch.randn(B, T, 4096)

    cs = ContentSynchronizer()
    C_s, C_v, kld_cs = cs(r_t)
    print(f"CS Normal: C_s={C_s.shape}, C_v={C_v.shape}")

    sd = StyleDisentangler()
    S_s, S_v, logits, kld_sd = sd(r_s, r_v)
    print(f"SD Normal: S_s={S_s.shape}, S_v={S_v.shape}")

    # Test 2: Flat shape (실제 에러 상황)
    r_t_flat = torch.randn(441, 4096)  # 21*21
    C_s_flat, C_v_flat, _ = cs(r_t_flat)
    print(f"CS Flat: C_s={C_s_flat.shape}, C_v={C_v_flat.shape}")
