
import torch, torch.nn as nn

class Reparam(nn.Module):
    def forward(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

def make_transformer(d_model=768, nhead=8, num_layers=4, dim_ff=2048, dropout=0.1, encoder=True):
    layer = nn.TransformerEncoderLayer if encoder else nn.TransformerDecoderLayer
    mod  = nn.TransformerEncoder if encoder else nn.TransformerDecoder
    l = layer(d_model, nhead, dim_ff, dropout, batch_first=True, norm_first=True)
    return mod(l, num_layers=num_layers)

class ContentSynchronizer(nn.Module):
    def __init__(self, d_in=768, d_latent=512, d_out=768, num_layers=4, nhead=8, dim_ff=2048, qdim=768):
        super().__init__()
        self.ffn_in = nn.Linear(d_in, d_out)
        self.enc = make_transformer(d_model=d_out, nhead=nhead, num_layers=num_layers, dim_ff=dim_ff, encoder=True)
        self.to_mu    = nn.Linear(d_out, d_latent)
        self.to_logvar= nn.Linear(d_out, d_latent)
        self.reparam  = Reparam()
        self.q_s_c = nn.Parameter(torch.randn(1, 1, qdim))
        self.q_v_c = nn.Parameter(torch.randn(1, 1, qdim))
        self.latent_to_mem = nn.Linear(d_latent, d_out)
        self.dec = make_transformer(d_model=d_out, nhead=nhead, num_layers=num_layers, dim_ff=dim_ff, encoder=False)
        self.proj_s = nn.Linear(d_out, d_out)
        self.proj_v = nn.Linear(d_out, d_out)

    def _decode(self, mem, q):
        B = mem.size(0)
        qB = q.expand(B, -1, -1)
        out = self.dec(tgt=qB, memory=mem)
        return out[:,0,:]

    def forward(self, r_t):
        x = self.ffn_in(r_t)
        h = self.enc(x)
        pooled = h.mean(dim=1)
        mu, lv = self.to_mu(pooled), self.to_logvar(pooled)
        z = self.reparam(mu, lv)
        mem = self.latent_to_mem(z).unsqueeze(1)
        C_s = self.proj_s(self._decode(mem, self.q_s_c))
        C_v = self.proj_v(self._decode(mem, self.q_v_c))
        kld = -0.5 * (1 + lv - mu.pow(2) - lv.exp()).mean()
        return C_s, C_v, kld

class StyleDisentangler(nn.Module):
    def __init__(self, d_in=768, d_latent=256, d_out=768, num_layers=4, nhead=8, dim_ff=2048, qdim=768,
                 n_emotions=7, n_age=4, n_gender=2, n_tone=3):
        super().__init__()
        self.ffn_s = nn.Linear(d_in, d_out)
        self.ffn_v = nn.Linear(d_in, d_out)
        self.enc_s = make_transformer(d_model=d_out, nhead=nhead, num_layers=num_layers, dim_ff=dim_ff, encoder=True)
        self.enc_v = make_transformer(d_model=d_out, nhead=nhead, num_layers=num_layers, dim_ff=dim_ff, encoder=True)
        self.q_s_e = nn.Parameter(torch.randn(1, 1, qdim))
        self.q_v_e = nn.Parameter(torch.randn(1, 1, qdim))
        self.q_s_p = nn.Parameter(torch.randn(1, 1, qdim))
        self.q_v_p = nn.Parameter(torch.randn(1, 1, qdim))
        self.to_mu_e = nn.Linear(d_out, d_latent); self.to_lv_e = nn.Linear(d_out, d_latent)
        self.to_mu_p = nn.Linear(d_out, d_latent); self.to_lv_p = nn.Linear(d_out, d_latent)
        self.reparam  = Reparam()
        self.latent_to_mem_e = nn.Linear(d_latent, d_out)
        self.latent_to_mem_p = nn.Linear(d_latent, d_out)
        self.dec = make_transformer(d_model=d_out, nhead=nhead, num_layers=num_layers, dim_ff=dim_ff, encoder=False)
        self.head_e = nn.Linear(d_out, d_out)
        self.head_p = nn.Linear(d_out, d_out)
        self.fuser_s = nn.Linear(d_out*2, d_out)
        self.fuser_v = nn.Linear(d_out*2, d_out)
        self.cls_emotion = nn.Linear(d_out, n_emotions)
        self.cls_age     = nn.Linear(d_out, n_age)
        self.cls_gender  = nn.Linear(d_out, n_gender)
        self.cls_tone    = nn.Linear(d_out, n_tone)

    def _encode_pool(self, enc, ffn, x):
        h = enc(ffn(x)); return h.mean(dim=1)

    def _decode_query(self, mem, q, proj):
        B = mem.size(0); qB = q.expand(B, -1, -1)
        out = self.dec(tgt=qB, memory=mem)
        return proj(out[:,0,:])

    def forward(self, r_s, r_v):
        hs = self._encode_pool(self.enc_s, self.ffn_s, r_s)
        hv = self._encode_pool(self.enc_v, self.ffn_v, r_v)
        mu_e_s, lv_e_s = self.to_mu_e(hs), self.to_lv_e(hs)
        mu_e_v, lv_e_v = self.to_mu_e(hv), self.to_lv_e(hv)
        z_e_s = self.reparam(mu_e_s, lv_e_s); z_e_v = self.reparam(mu_e_v, lv_e_v)
        mem_e_s = self.latent_to_mem_e(z_e_s).unsqueeze(1); mem_e_v = self.latent_to_mem_e(z_e_v).unsqueeze(1)
        E_s = self._decode_query(mem_e_s, self.q_s_e, self.head_e)
        E_v = self._decode_query(mem_e_v, self.q_v_e, self.head_e)
        mu_p_s, lv_p_s = self.to_mu_p(hs), self.to_lv_p(hs)
        mu_p_v, lv_p_v = self.to_mu_p(hv), self.to_lv_p(hv)
        z_p_s = self.reparam(mu_p_s, lv_p_s); z_p_v = self.reparam(mu_p_v, lv_p_v)
        mem_p_s = self.latent_to_mem_p(z_p_s).unsqueeze(1); mem_p_v = self.latent_to_mem_p(z_p_v).unsqueeze(1)
        P_s = self._decode_query(mem_p_s, self.q_s_p, self.head_p)
        P_v = self._decode_query(mem_p_v, self.q_v_p, self.head_p)
        S_s = self.fuser_s(torch.tanh(torch.cat([E_s, P_s], dim=-1)))
        S_v = self.fuser_v(torch.tanh(torch.cat([E_v, P_v], dim=-1)))
        E_global = 0.5*(E_s + E_v); P_global = 0.5*(P_s + P_v)
        logits = dict(emotion=self.cls_emotion(E_global),
                      age=self.cls_age(P_global),
                      gender=self.cls_gender(P_global),
                      tone=self.cls_tone(P_global))
        kld = 0.0
        for mu, lv in [(mu_e_s, lv_e_s),(mu_e_v, lv_e_v),(mu_p_s, lv_p_s),(mu_p_v, lv_p_v)]:
            kld = kld + (-0.5 * (1 + lv - mu.pow(2) - lv.exp()).mean())
        return S_s, S_v, logits, kld
