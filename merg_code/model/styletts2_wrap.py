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
from StyleTTS2.models import ProsodyPredictor
from StyleTTS2.Modules.hifigan import Generator, Decoder
from StyleTTS2.models import build_model


class PLBERTWrapper(nn.Module):
    def __init__(self, plbert_dir, device=None):
        """
        plbert_dir 안에 있어야 할 것:
          - config.yml        (dataset_params.tokenizer, token_maps 등)
          - token_maps.pkl    (phoneme/token -> id 맵)
          - step_*.t7         (PL-BERT checkpoint, 예: step_1000000.t7)
        """
        super().__init__()
        self.plbert_dir = plbert_dir
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1) config.yml 로드
        cfg_path = os.path.join(plbert_dir, "config.yml")
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
        dset_cfg = cfg["dataset_params"]
        self.token_sep = dset_cfg.get("token_separator", " ")
        self.token_mask = dset_cfg.get("token_mask", "M")
        self.word_sep_id = dset_cfg.get("word_separator", 3039)
        base_tok_name = dset_cfg["tokenizer"]  # 보통 "transfo-xl-wt103"[web:70]

        # 2) base tokenizer 로드 (Transformer-XL)
        self.base_tokenizer = TransfoXLTokenizer.from_pretrained(base_tok_name)

        # 3) token_maps.pkl 로드 (phoneme/token -> vocab id)
        tmaps_path = os.path.join(plbert_dir, dset_cfg["token_maps"])  # "token_maps.pkl"
        with open(tmaps_path, "rb") as f:
            self.token_maps = pickle.load(f)
        # self.token_maps: {str_token: int_id, ...}

        # 4) PL-BERT 모델 로드 (TransfoXLModel + step_*.t7 state_dict)
        self.plbert = TransfoXLModel.from_pretrained(base_tok_name)

        ckpt_name = None
        for fn in os.listdir(plbert_dir):
            if fn.startswith("step_") and fn.endswith(".t7"):
                ckpt_name = fn
                break
        if ckpt_name is None:
            raise FileNotFoundError(f"No step_*.t7 found under {plbert_dir}")
        ckpt = torch.load(os.path.join(plbert_dir, ckpt_name), map_location="cpu")
        state_dict = ckpt.get("net", ckpt)
        self.plbert.load_state_dict(state_dict, strict=False)

        self.plbert.to(self.device)
        self.plbert.eval()
        self.plbert.requires_grad_(False)

    def _tokens_to_ids(self, tokens):
        """
        tokens: List[str] (이미 token_separator 기준으로 split된 토큰들)
        우선 token_maps를 적용, 없으면 base tokenizer로 encode.
        """
        ids = []
        for tok in tokens:
            if tok in self.token_maps:
                ids.append(self.token_maps[tok])
            else:
                sub_ids = self.base_tokenizer.encode(tok, add_special_tokens=False)
                ids.extend(sub_ids)
        # 문장 끝에 word_separator id 추가
        ids.append(self.word_sep_id)
        return ids

    @torch.no_grad()
    def encode_texts(self, texts):
        """
        texts: List[str] 또는 str
        - 이상적으로는 phoneme 시퀀스를 token_separator 로 join한 문자열.
        - 현재는 response 문장을 그대로 넣되, token_separator 기준으로 split해서 token_maps 적용.
        return:
          - hidden_states: (B, T, H)
        """
        if isinstance(texts, str):
            texts = [texts]

        all_ids = []
        max_len = 0
        for txt in texts:
            tokens = txt.split(self.token_sep)  # config.yml 의 token_separator 사용
            ids = self._tokens_to_ids(tokens)
            all_ids.append(ids)
            max_len = max(max_len, len(ids))

        pad_id = self.base_tokenizer.pad_token_id or 0
        input_ids = []
        for ids in all_ids:
            pad_len = max_len - len(ids)
            input_ids.append(ids + [pad_id] * pad_len)
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device)  # (B, T)

        out = self.plbert(input_ids=input_ids)
        h = out.last_hidden_state  # (B, T, H)
        return h


class StyleTTS2Encoders(nn.Module):
    def __init__(self, ckpt_dir, proj_dim=768, device=None, target_sr=24000):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_sr = target_sr

        # 1) PLBERT + tokenizer + token_maps
        plbert_dir = 'merg_code/StyleTTS2/Utils/PLBERT'
        self.plbert_wrap = PLBERTWrapper(plbert_dir, device=self.device)
        self.text_aco = self.plbert_wrap.plbert  # PL-BERT 본체

        hidden_dim = self.text_aco.config.d_model  # Transformer-XL hidden size
        self.text_proj = nn.Linear(hidden_dim, proj_dim, bias=False).to(self.device)

        # (선택) MERG 쪽 finetune weight 반영
        text_aco_ckpt_path = os.path.join(ckpt_dir, 'text_aco_encoder.pt')
        if os.path.exists(text_aco_ckpt_path):
            text_aco_ckpt = torch.load(text_aco_ckpt_path, map_location='cpu')
            self.text_aco.load_state_dict(text_aco_ckpt.get('model', {}), strict=False)

        # 2) JDCNet 스타일 인코더 (E_ref)
        '''
            wav file (24kHz) 
                ↓ torchaudio.load + resample(24000)
            waveform tensor (B, T) 
                ↓ MelSpectrogram(n_fft=1024, hop=256, n_mels=80)
            mel_spec (B, 80, T') 
                ↓ log1p + 길이 192로 crop/pad 
            mel_192 (B, 80, 192) 
                ↓ unsqueeze(1) 
            mel_jdc (B, 1, 80, 192)  ← JDCNet(seq_len=192) 입력
                ↓ JDCNet.forward()
            F0_real (B, 192)          ← style vector 완성! 
        '''
        self.ref_enc = JDCNet(num_class=1, seq_len=192)
        ref_enc_ckpt_path = os.path.join(ckpt_dir, 'reference_encoder.pt')
        if os.path.exists(ref_enc_ckpt_path):
            ref_enc_ckpt = torch.load(ref_enc_ckpt_path, map_location='cpu')
            jdc_state = ref_enc_ckpt.get('net', {})
            jdc_state.pop('classifier.weight', None)
            jdc_state.pop('classifier.bias', None)
            self.ref_enc.load_state_dict(jdc_state, strict=False)

        self.text_aco.to(self.device).eval().requires_grad_(False)
        self.text_proj.eval().requires_grad_(False)
        self.ref_enc.to(self.device).eval().requires_grad_(False)

    # ------------------ 텍스트 인코더 ------------------
    @torch.no_grad()
    def text_content(self, texts):
        """
        texts: List[str] 또는 str (Empatheia의 response 텍스트)
        return: (B, proj_dim)
        """
        h_text = self.plbert_wrap.encode_texts(texts)  # (B, T, H)
        h_mean = h_text.mean(dim=1)  # (B, H)
        h_proj = self.text_proj(h_mean)  # (B, proj_dim)
        return h_proj

    # ------------------ 오디오 로더 ------------------
    @torch.no_grad()
    def _load_wav(self, path):
        """
        path: str (.wav 경로)
        return: 1D waveform (T,) @ target_sr, mono
        """
        wav, sr = torchaudio.load(path)
        # mono
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        # resample
        if sr != self.target_sr:
            wav = torchaudio.functional.resample(wav, sr, self.target_sr)
        wav = wav.squeeze(0)  # (T,)
        return wav

    @torch.no_grad()
    def _paths_to_batch(self, paths):
        """
        paths: List[str]
        return: (B, T) waveform batch @ target_sr
        """
        if not paths:
            raise ValueError("paths cannot be empty.")

        wavs = [self._load_wav(p) for p in paths]  # list of (T_i,)
        max_len = max(w.size(0) for w in wavs)
        batch = []
        for w in wavs:
            pad_len = max_len - w.size(0)
            if pad_len > 0:
                w = F.pad(w, (0, pad_len))  # 뒤쪽 zero-pad
            batch.append(w)
        batch = torch.stack(batch, dim=0)  # (B, T)
        return batch.to(self.device)

    # ------------------ JDC 입력 변환 ------------------
    @torch.no_grad()
    def _wav_to_jdc_input(self, wav):
        """
        StyleTTS2 원본 preprocess(wave) 재현
        """
        device = wav.device

        # 원본 MEL_PARAMS (StyleTTS2 표준)
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=24000,
            n_fft=1024,
            hop_length=256,  # hop*2=512ms frame
            win_length=1024,
            n_mels=80,
            f_min=0.0,
            f_max=8000.0,
            power=1.0
        ).to(device)

        mel = mel_transform(wav)  # (B, 80, T')
        mel = torch.log1p(mel)  # 원본 preprocess

        # 길이 192로 맞추기 (seq_len=192)
        B, F, T = mel.shape
        if T > 192:
            start = (T - 192) // 2
            mel = mel[:, :, start:start + 192]
        elif T < 192:
            pad_t = 192 - T
            mel = torch.nn.functional.pad(mel, (0, pad_t))

        # JDCNet 입력 형태: (B, 1, 80, 192)
        mel_jdc = mel.unsqueeze(1)  # (B, 1, 80, 192)

        return mel_jdc

    # ------------------ 스타일 추출 ------------------
    @torch.no_grad()
    def style_from_audio(self, wav_inputs):
        if isinstance(wav_inputs, str):
            wav_inputs = [wav_inputs]
        if isinstance(wav_inputs, (list, tuple)):
            wav_batch = self._paths_to_batch(wav_inputs)
        else:
            wav_batch = wav_inputs.to(self.device)

        mel_jdc = self._wav_to_jdc_input(wav_batch)  # (B, 1, 80, 192)

        # 원본 pitch_extractor와 동일
        F0_real, _, F0 = self.ref_enc(mel_jdc)  # F0_real = style feature

        # ★ F0_real 그대로 사용 (이미 평균된 상태)
        return F0_real  # (B, C) 형태로 나옴

class StyleTTS2Decoders(nn.Module):
    """
    CS / SD output을 직접 받아 StyleTTS2 hifigan Decoder로 waveform 생성
    - TextEncoder ❌
    - StyleEncoder ❌
    - ProsodyPredictor ❌
    """

    def __init__(self, styletts2_ckpt_path, device="cuda"):
        """
        styletts2_ckpt_path:
            StyleTTS2 pretrained checkpoint (.pth / .pt)
            내부에 'decoder', 'model_params.decoder'가 있어야 함
        """
        super().__init__()
        self.device = device

        ckpt = torch.load(styletts2_ckpt_path, map_location="cpu")

        # StyleTTS2 hifigan Decoder
        self.decoder = Decoder(**ckpt["model_params"]["decoder"])
        self.decoder.load_state_dict(ckpt["decoder"])
        self.decoder.to(device).eval()

    @torch.no_grad()
    def forward(self, C_s, S_s, F0=None, N=None):
        """
        Args:
            C_s: (B, C, T)    ← CS output (frame-level content)
            S_s: (B, D)       ← SD output (style embedding)
            F0:  (B, T) or None
            N:   (B, T) or None

        Returns:
            wav: (B, 1, T_audio)
        """
        C_s = C_s.to(self.device)
        S_s = S_s.to(self.device)

        B, _, T = C_s.shape

        # StyleTTS2는 F0 / N 필수 → 없으면 0으로 대체
        if F0 is None:
            F0 = torch.zeros(B, T, device=self.device)
        else:
            F0 = F0.to(self.device)

        if N is None:
            N = torch.zeros(B, T, device=self.device)
        else:
            N = N.to(self.device)

        # hifigan Decoder
        wav = self.decoder(
            asr=C_s,
            F0_curve=F0,
            N=N,
            s=S_s
        )

        return wav


if __name__=='__main__':
    # test code
    sty = StyleTTS2Encoders('/home/a6000/bk-project/multimodal-Empatheia/ckpt/pretrained_ckpt/styletts2_encoders')
    S_s_gold = sty.style_from_audio('/mnt/dataset/AvaMERG_jhchoi/AvaMERG/audio_v5_0/dia14724utt3_18.wav')
    print(f"S_s_gold shape: {S_s_gold.shape}")  # (1, C)
    print(S_s_gold)
