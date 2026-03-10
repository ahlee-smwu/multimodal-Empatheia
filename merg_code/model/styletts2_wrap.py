import torch
import torch.nn as nn
import os
import sys
import yaml
import pickle
from transformers import TransfoXLTokenizer, TransfoXLModel
import torchaudio
import torch.nn.functional as F

sys.path.append('merg_code/StyleTTS2')

from StyleTTS2.Utils.PLBERT.util import load_plbert
from StyleTTS2.Utils.JDC.model import JDCNet
from StyleTTS2.models import ProsodyPredictor, StyleEncoder
from StyleTTS2.Modules.hifigan import Generator, Decoder
from StyleTTS2.models import build_model
from Modules.hifigan import Decoder
import yaml
from munch import Munch
from transformers import AlbertModel, AlbertTokenizer

'''
Text
 └─ TextEncoder(PL-BERT) → C_token (B, H, T_token)

Reference
 └─ StyleEncoder(JDCNet) → S (B, D)

(C_token, S)
 └─ ProsodyPredictor  # in .pth weight
      ├─ duration (B, T_token)
      ├─ F0_token (B, T_token)
      └─ Energy_token (B, T_token)

      ↓

Length Regulator (duration 기반 확장)
      ↓

C_frame (B, H, T_frame)
F0_frame (B, T_frame)
Energy_frame (B, T_frame)

(C_frame, S, F0_frame, Energy_frame)
 └─ HiFiGAN Decoder # in .pth weight
      ↓
Waveform
'''

class PLBERTWrapper(nn.Module):
    def __init__(self, plbert_dir, device=None):
        super().__init__()

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.plbert_dir = plbert_dir

        # ----------------------------
        # 1. config 로드
        # ----------------------------
        with open(os.path.join(plbert_dir, "config.yml"), "r") as f:
            cfg = yaml.safe_load(f)

        dset_cfg = cfg["dataset_params"]

        self.token_sep = dset_cfg.get("token_separator", " ")
        self.word_sep_id = dset_cfg.get("word_separator", 3039)

        # ----------------------------
        # 2. token_maps 로드
        # ----------------------------
        with open(os.path.join(plbert_dir, dset_cfg["token_maps"]), "rb") as f:
            self.token_maps = pickle.load(f)
        # print(type(self.token_maps))
        # print(list(self.token_maps.items())[:10])

        # ----------------------------
        # 3. Custom PLBERT 로드
        # ----------------------------
        self.plbert = load_plbert(plbert_dir)

        self.plbert.to(self.device)
        self.plbert.eval()
        self.plbert.requires_grad_(False)

    # ----------------------------
    # text → ids
    # ----------------------------
    def _tokens_to_ids(self, tokens):
        ids = []
        for tok in tokens:
            if tok in self.token_maps:
                ids.append(self.token_maps[tok])
        ids.append(self.word_sep_id)
        return ids

    # ----------------------------
    # encode
    # ----------------------------
    @torch.no_grad()
    def encode_texts(self, texts):

        if isinstance(texts, str):
            texts = [texts]

        all_ids = []
        max_len = 0

        for txt in texts:

            words = txt.lower().split()

            ids = []

            for w in words:
                if w in self.token_maps:
                    ids.append(self.token_maps[w])
                else:
                    # OOV는 0으로
                    ids.append(0)

            if len(ids) == 0:
                ids = [0]

            all_ids.append(ids)
            max_len = max(max_len, len(ids))

        # padding
        input_ids = []
        for ids in all_ids:
            pad_len = max_len - len(ids)
            input_ids.append(ids + [0] * pad_len)

        input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device)

        out = self.plbert(input_ids=input_ids)
        return out #(B,T,768)

class StyleTTS2Encoders(nn.Module):
    def __init__(self, ckpt_path, device=None, target_sr=24000):
        super().__init__()

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_sr = target_sr

        # ----------------------------
        # 1. config 로드
        # ----------------------------
        with open(os.path.join(ckpt_path, "config.yml"), "r") as f:
            config = yaml.safe_load(f)

        mp = config["model_params"]

        hidden_dim = mp["hidden_dim"]   # 512
        style_dim = mp["style_dim"]     # 128

        # ----------------------------
        # 2. checkpoint 먼저 로드
        # ----------------------------
        ckpt = torch.load(
            os.path.join(ckpt_path, "epochs_2nd_00020.pth"),
            map_location="cpu"
        )
        net = ckpt["net"]

        # ----------------------------
        # 3. PLBERT 로드 (Custom)
        # ----------------------------
        plbert_dir = 'merg_code/StyleTTS2/Utils/PLBERT'
        self.plbert_wrap = PLBERTWrapper(plbert_dir, device=self.device)

        t_weight = net["bert_encoder"]["module.weight"]  # (512,768)
        t_bias = net["bert_encoder"]["module.bias"]

        self.bert_encoder = nn.Linear(
            t_weight.size(1),  # 768
            t_weight.size(0)  # 512
        )

        self.bert_encoder.weight.data.copy_(t_weight)
        self.bert_encoder.bias.data.copy_(t_bias)

        # ----------------------------
        # 4. Style Encoder (Acoustic)
        # ----------------------------
        from StyleTTS2.models import StyleEncoder

        self.style_encoder = StyleEncoder(
            dim_in=mp["dim_in"],
            style_dim=style_dim,
            max_conv_dim=hidden_dim
        )

        style_state = {
            k.replace("module.", ""): v
            for k, v in net["style_encoder"].items()
        }
        self.style_encoder.load_state_dict(style_state, strict=True)

        # ----------------------------
        # 5. Predictor Encoder (Prosodic)
        # ----------------------------
        self.predictor_encoder = StyleEncoder(
            dim_in=mp["dim_in"],
            style_dim=style_dim,
            max_conv_dim=hidden_dim
        )

        predictor_enc_state = {
            k.replace("module.", ""): v
            for k, v in net["predictor_encoder"].items()
        }
        self.predictor_encoder.load_state_dict(predictor_enc_state, strict=True)

        # ----------------------------
        # 6. freeze
        # ----------------------------
        self.to(self.device)
        self.eval()
        self.requires_grad_(False)

    # --------------------------------
    # TEXT → C_token (B,512,T)
    # --------------------------------
    @torch.no_grad()
    def text_content(self, texts):

        h = self.plbert_wrap.encode_texts(texts)  # (B,T,768)

        if h.size(-1) != self.bert_encoder.in_features:
            raise RuntimeError(
                f"Hidden mismatch: {h.size(-1)} vs {self.bert_encoder.in_features}"
            )

        h = self.bert_encoder(h)  # (B,T,512)

        return h.transpose(1, 2)  # (B,512,T)

    @torch.no_grad()
    def wav_to_mel(self, wav):
        """
        wav: (B,T) or (T,)
        return: (B,80,T')
        """

        if wav.dim() == 1:
            wav = wav.unsqueeze(0)

        wav = wav.to(self.device)

        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=24000,
            n_fft=2048,
            win_length=1200,
            hop_length=300,
            n_mels=80,
            f_min=0.0,
            f_max=8000.0,
            power=1.0,
        ).to(self.device)

        mel = mel_transform(wav)  # (B,80,T')
        mel = torch.log(mel + 1e-5)

        return mel

    # --------------------------------
    # MEL → Style (128, 128)
    # --------------------------------
    @torch.no_grad()
    def style_from_audio(self, wav):
        try:
            mel = self.wav_to_mel(wav)
            # -----------------------------
            # 1) shape 정리 (B,1,80,T)
            # -----------------------------
            if mel.dim() == 2:
                mel = mel.unsqueeze(0).unsqueeze(0)
            elif mel.dim() == 3:
                mel = mel.unsqueeze(1)
            elif mel.dim() == 4:
                if mel.size(1) != 1:
                    mel = mel[:, :1]
            else:
                raise RuntimeError(f"Unexpected mel shape: {mel.shape}")

            # -----------------------------
            # 2) 너무 짧으면 바로 dummy 반환
            # -----------------------------
            if mel.size(-1) < 10 or mel.size(-2) < 10:
                B = mel.size(0)
                device = mel.device
                return torch.zeros(B, 128, device=device)

            # -----------------------------
            # 3) 실제 forward
            # -----------------------------
            s_acoustic = self.style_encoder(mel)
            # s_prosodic = self.predictor_encoder(mel)

            return s_acoustic #, s_prosodic

        except Exception as e:
            print(f"[WARN] style_from_audio fallback: {e}")
            if isinstance(wav, torch.Tensor):
                B = wav.size(0) if wav.dim() > 1 else 1
                device = wav.device
            else:
                B = 1
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            return torch.zeros(B, 128, device=device)

class StyleTTS2Decoders(nn.Module):
    def __init__(self, ckpt_path, device="cuda"):
        super().__init__()
        self.device = device

        with open(os.path.join(ckpt_path, "config.yml"), "r") as f:
            config = yaml.safe_load(f)

        mp = config["model_params"]
        self.hidden_dim = mp["hidden_dim"]   # 512
        self.style_dim  = mp["style_dim"]    # 128
        self.max_dur    = mp["max_dur"]      # 여기 값이 15인 경우가 많음

        self.prosody = ProsodyPredictor(
            style_dim=self.style_dim,
            d_hid=self.hidden_dim,
            nlayers=mp["n_layer"],
            max_dur=self.max_dur,
            dropout=mp["dropout"],
        )

        self.decoder = Decoder(
            dim_in=self.hidden_dim,
            style_dim=self.style_dim,
            dim_out=mp["n_mels"],
            resblock_kernel_sizes=mp["decoder"]["resblock_kernel_sizes"],
            upsample_rates=mp["decoder"]["upsample_rates"],
            upsample_initial_channel=mp["decoder"]["upsample_initial_channel"],
            resblock_dilation_sizes=mp["decoder"]["resblock_dilation_sizes"],
            upsample_kernel_sizes=mp["decoder"]["upsample_kernel_sizes"],
        )

        ckpt = torch.load(os.path.join(ckpt_path, "epochs_2nd_00020.pth"), map_location="cpu")
        net = ckpt["net"]

        predictor_state = {k.replace("module.", ""): v for k, v in net["predictor"].items()}
        self.prosody.load_state_dict(predictor_state, strict=True)

        decoder_state = {k.replace("module.", ""): v for k, v in net["decoder"].items()}
        self.decoder.load_state_dict(decoder_state, strict=True)

        self.to(device)
        self.eval()
        self.requires_grad_(False)

    @torch.no_grad()
    def forward(self, C_s, S_s):
        """
        C_s: (B, 512, T_text)  e.g. (1,512,15)
        S_s: (B, 128)          e.g. (1,128)
        return: wav
        """

        # ============================================================
        # 0) 입력 정규화
        # ============================================================
        C_token = C_s
        S = S_s

        if C_token.dim() == 2:
            C_token = C_token.unsqueeze(0)
        if S.dim() == 1:
            S = S.unsqueeze(0)
        if S.dim() == 3 and S.size(1) == 1:
            S = S.squeeze(1)

        # (B,T,512) -> (B,512,T)
        if C_token.size(1) != 512 and C_token.size(-1) == 512:
            C_token = C_token.transpose(1, 2)

        C_token = C_token.to(self.device)
        S = S.to(self.device)

        B, Cc, T_text = C_token.shape
        if Cc != 512:
            raise RuntimeError(f"[0] C_token must be (B,512,T_text), got {tuple(C_token.shape)}")
        if S.shape != (B, 128):
            raise RuntimeError(f"[0] S must be (B,128), got {tuple(S.shape)}")

        text_lengths = torch.full((B,), T_text, dtype=torch.long, device=self.device)
        mask = torch.zeros((B, T_text), dtype=torch.bool, device=self.device)

        # ============================================================
        # 1) dummy alignment
        # ============================================================
        alignment = torch.eye(T_text, device=self.device).unsqueeze(0).expand(B, -1, -1)

        # ============================================================
        # 2) duration logits -> duration
        #    관측: duration_logits: (B, T_text, Nd=50)
        # ============================================================
        duration_logits, _ = self.prosody(C_token, S, text_lengths, alignment, mask)

        if duration_logits.dim() != 3 or duration_logits.size(0) != B or duration_logits.size(1) != T_text:
            raise RuntimeError(
                f"[2] duration_logits unexpected shape {tuple(duration_logits.shape)}; "
                f"expected (B,T_text,Nd)=({B},{T_text},Nd)"
            )

        # (B,T_text,Nd) -> (B,T_text) duration (>=1)
        duration = torch.argmax(duration_logits, dim=-1).long() + 1
        duration = duration.clamp(min=1)

        # ============================================================
        # 3) length regulator: token -> frame
        #    C_frame: (B, T_asr, 512)
        # ============================================================
        x_tok = C_token.transpose(1, 2)  # (B,T_text,512)
        C_frame = self._length_regulator(x_tok, duration)  # (B,T_asr,512)

        if C_frame.dim() != 3 or C_frame.size(0) != B or C_frame.size(2) != 512:
            raise RuntimeError(f"[3] C_frame must be (B,T_asr,512), got {tuple(C_frame.shape)}")

        T_asr = C_frame.size(1)
        C_frame_ct = C_frame.transpose(1, 2)  # (B,512,T_asr)

        # ============================================================
        # 4) F0/N predictor 입력: (B,640,T_asr) = (512+128,T)
        # ============================================================
        S_ct = S.unsqueeze(-1).expand(B, 128, T_asr)  # (B,128,T_asr)
        F0N_in = torch.cat([C_frame_ct, S_ct], dim=1)  # (B,640,T_asr)

        F0_frame, N_frame = self.prosody.F0Ntrain(F0N_in, S)

        # ============================================================
        # 5) ✅ HiFiGAN(decoder) 규약에 맞추기
        #    - decoder는 F0_curve를 (B,T)로 받음 (내부에서 unsqueeze(1))
        #    - N도 (B,T)
        #    - asr은 (B,512,T)
        #    - T가 다르면 정수배 repeat로만 맞춤 (보간 금지)
        # ============================================================

        # decoder 규약: F0_curve, N은 (B,T)로 넣는다 (decoder가 내부에서 unsqueeze(1)함)
        F0_curve = self._to_BT_strict(F0_frame, name="F0_curve")  # (B,T_f0)
        N_curve = self._to_BT_strict(N_frame, name="N")  # (B,T_n)

        T_f0 = F0_curve.size(1)
        T_n = N_curve.size(1)
        if T_f0 != T_n:
            raise RuntimeError(f"[final] F0/N time mismatch: F0={T_f0}, N={T_n}")

        # asr은 (B,512,T_asr)
        asr = C_frame.transpose(1, 2)  # (B,512,T_asr)
        T_asr = asr.size(2)

        # --- 정수배 repeat로 시간축 정합 (양방향 모두 처리) ---
        if T_asr != T_f0:
            if T_asr % T_f0 == 0:
                # F0/N이 더 짧다 -> F0/N을 늘린다 (지금 네 에러가 이 케이스: 30 vs 15)
                r = T_asr // T_f0
                F0_curve = torch.repeat_interleave(F0_curve, repeats=r, dim=1)  # (B,T_asr)
                N_curve = torch.repeat_interleave(N_curve, repeats=r, dim=1)  # (B,T_asr)

            elif T_f0 % T_asr == 0:
                # asr이 더 짧다 -> asr을 늘린다
                r = T_f0 // T_asr
                asr = torch.repeat_interleave(asr, repeats=r, dim=2)  # (B,512,T_f0)
                T_asr = asr.size(2)

            else:
                raise RuntimeError(
                    f"[final] Cannot match time by integer repeat: T_asr={T_asr}, T_f0={T_f0}"
                )

        # 최종 검증: decoder cat이 절대 안 터지게(규약대로) 확인
        T = asr.size(2)
        if F0_curve.size(1) != T or N_curve.size(1) != T:
            raise RuntimeError(
                f"[final] Time mismatch before decoder: asr T={T}, F0 T={F0_curve.size(1)}, N T={N_curve.size(1)}"
            )
        if asr.size(1) != 512:
            raise RuntimeError(f"[final] asr channel mismatch: expected 512, got {asr.size(1)}")

        # decoder 호출
        # print(asr.shape, F0_curve.shape, N_curve.shape, S.shape)
        # ars:torch.Size([1, 512, 30]), FO:torch.Size([1, 30]), N:torch.Size([1, 30]), s:torch.Size([1, 128])

        # --- decoder 내부 conv를 실제로 돌려서, cat 직전의 길이를 맞춘다 (정확) ---
        with torch.no_grad():
            # decoder 안에서 만들어질 F0/N feature의 time 길이를 미리 확인
            F0_feat_T = self.decoder.F0_conv(F0_curve.unsqueeze(1)).size(-1)
            N_feat_T = self.decoder.N_conv(N_curve.unsqueeze(1)).size(-1)

        T_asr = asr.size(2)

        # F0_conv와 N_conv는 같은 downsample을 해야 정상
        if F0_feat_T != N_feat_T:
            raise RuntimeError(f"Decoder conv time mismatch: F0_feat_T={F0_feat_T}, N_feat_T={N_feat_T}")

        # 1) conv 결과가 asr보다 짧으면: F0/N 입력 길이를 늘려서(conv 후) asr와 같게 만든다
        if F0_feat_T < T_asr:
            if T_asr % F0_feat_T != 0:
                raise RuntimeError(f"Cannot integer-match conv output to asr: T_asr={T_asr}, F0_feat_T={F0_feat_T}")

            r = T_asr // F0_feat_T  # 네 케이스는 보통 2 (30 // 15)
            # F0_curve/N_curve의 입력 time을 r배로 늘리면, stride=2인 conv면 출력이 r배 늘어남
            F0_curve = torch.repeat_interleave(F0_curve, repeats=r, dim=1)
            N_curve = torch.repeat_interleave(N_curve, repeats=r, dim=1)

            # 재확인
            with torch.no_grad():
                F0_feat_T2 = self.decoder.F0_conv(F0_curve.unsqueeze(1)).size(-1)
                N_feat_T2 = self.decoder.N_conv(N_curve.unsqueeze(1)).size(-1)
            if F0_feat_T2 != T_asr or N_feat_T2 != T_asr:
                raise RuntimeError(
                    f"After upsample, conv output still mismatch: "
                    f"F0_feat_T={F0_feat_T2}, N_feat_T={N_feat_T2}, asr_T={T_asr}"
                )

        # 2) conv 결과가 asr보다 길면: asr을 줄여서 맞춘다 (이 케이스는 드물지만 정석으로 처리)
        elif F0_feat_T > T_asr:
            if F0_feat_T % T_asr != 0:
                raise RuntimeError(f"Cannot integer-match asr to conv output: T_asr={T_asr}, F0_feat_T={F0_feat_T}")
            r = F0_feat_T // T_asr
            # 시간축을 r배로 줄임(정수배) - 보간 금지
            asr = asr[:, :, ::r]

            # 최종 확인: asr 길이 == conv 결과 길이
            if asr.size(2) != F0_feat_T:
                raise RuntimeError(f"After downsample asr, mismatch: asr_T={asr.size(2)}, conv_T={F0_feat_T}")

        # 같으면 그대로 진행
        # print("FINAL shapes -> asr", asr.shape, "F0_curve", F0_curve.shape, "N", N_curve.shape)

        wav = self.decoder(
            asr=asr,  # (B,512,T)
            F0_curve=F0_curve,  # (B,T)
            N=N_curve,  # (B,T)
            s=S
        )
        return wav

    # ------------------------------------------------------------
    # helper: (B,T)로 엄격 변환 (decoder 규약)
    # ------------------------------------------------------------
    @torch.no_grad()
    def _to_BT_strict(self, x, name="tensor"):
        """
        decoder가 기대하는 (B,T)로 엄격 변환.
        - (B,T) -> OK
        - (B,1,T) -> squeeze(1)
        - (B,T,1) -> squeeze(-1)
        그 외 -> 에러 (보간/흉내 금지)
        """
        if x.dim() == 2:
            return x
        if x.dim() == 3:
            if x.size(1) == 1:   # (B,1,T)
                return x.squeeze(1)
            if x.size(-1) == 1:  # (B,T,1)
                return x.squeeze(-1)
        raise RuntimeError(f"{name} must be (B,T) or squeezable, got {tuple(x.shape)}")

    # ------------------------------------------------------------
    # helper: length regulator
    # ------------------------------------------------------------
    @torch.no_grad()
    def _length_regulator(self, x, durations):
        """
        x: (B, T_text, 512)
        durations: (B, T_text) positive ints
        return: (B, T_asr, 512)
        """
        if durations.dim() != 2:
            durations = durations.view(durations.size(0), -1)

        B, T_text, C = x.shape
        if C != 512:
            raise RuntimeError(f"x must have C=512, got {C}")
        if durations.shape != (B, T_text):
            raise RuntimeError(f"durations must be (B,T_text)=({B},{T_text}), got {tuple(durations.shape)}")

        expanded = []
        for b in range(B):
            reps = durations[b].long().view(-1)    # (T_text,)
            x_b = x[b]                             # (T_text,512)
            x_exp = torch.repeat_interleave(x_b, reps, dim=0)
            expanded.append(x_exp)

        max_len = max(e.size(0) for e in expanded)
        padded = [F.pad(e, (0, 0, 0, max_len - e.size(0))) for e in expanded]
        return torch.stack(padded, dim=0)

if __name__=='__main__':
    # test code
    sty = StyleTTS2Encoders('/home/a6000/bk-project/multimodal-Empatheia/ckpt/pretrained_ckpt/styletts2_encoders')
    S_s_gold = sty.style_from_audio('/mnt/dataset/AvaMERG_jhchoi/AvaMERG/audio_v5_0/dia14724utt3_18.wav')
    print(f"S_s_gold shape: {S_s_gold.shape}")  # (1, C)
    print(S_s_gold)
