import torch
import torch.nn as nn
import os
import sys

# 이 파일이 merg_code/model/styletts2_wrap.py 라고 가정
sys.path.append('merg_code/StyleTTS2')

from StyleTTS2.Utils.PLBERT.util import load_plbert
from StyleTTS2.Utils.JDC.model import JDCNet


class StyleTTS2Encoders(nn.Module):
    def __init__(self, ckpt_dir, proj_dim=768):
        super().__init__()

        # 1) 외부 ckpt 로드 (MERG 쪽에서 만든 것)
        text_aco_ckpt = torch.load(os.path.join(ckpt_dir, 'text_aco_encoder.pt'),
                                   map_location='cpu')
        text_bert_ckpt = torch.load(os.path.join(ckpt_dir, 'text_bert_encoder.pt'),
                                    map_location='cpu')
        ref_enc_ckpt = torch.load(os.path.join(ckpt_dir, 'reference_encoder.pt'),
                                  map_location='cpu')

        # 2) PLBERT 텍스트 인코더 (E_aco)
        plbert_dir = 'merg_code/StyleTTS2/Utils/PLBERT'
        self.text_bert = load_plbert(plbert_dir)
        self.text_aco  = load_plbert(plbert_dir)
        hidden_dim = self.text_aco.config.hidden_size  # 보통 768

        # projection 레이어 정의 (H -> proj_dim)
        self.text_proj = nn.Linear(hidden_dim, proj_dim, bias=False)

        # MERG 쪽 finetune weight 반영 (있으면)
        self.text_bert.load_state_dict(text_bert_ckpt.get('net', {}), strict=False)
        self.text_aco.load_state_dict(text_aco_ckpt.get('model', {}), strict=False)

        # 3) JDCNet 스타일 인코더 (E_ref)
        # ckpt는 num_class=1 로 학습됐다고 보고 classifier는 버림
        self.ref_enc = JDCNet(num_class=1)
        jdc_state = ref_enc_ckpt.get('net', {})
        jdc_state.pop('classifier.weight', None)
        jdc_state.pop('classifier.bias', None)
        self.ref_enc.load_state_dict(jdc_state, strict=False)

        for m in [self.text_aco, self.text_bert, self.ref_enc]:
            m.eval()
            m.requires_grad_(False)

    @torch.no_grad()
    def text_content(self, input_ids, attention_mask=None):
        """
        input_ids      : (B, T), tokenizer가 만든 토큰 id
        attention_mask : (B, T) or None
        return         : (B, proj_dim)
        """
        # PLBERT: last_hidden_state (B, T, H)
        h_text = self.text_aco(input_ids=input_ids,
                               attention_mask=attention_mask)  # (B, T, H)

        # time 평균 → (B, H)
        h_mean = h_text.mean(dim=1)

        # projection → (B, proj_dim)
        h_proj = self.text_proj(h_mean)
        return h_proj

    @torch.no_grad()
    def _wav_to_jdc_input(self, wav):
        """
        wav: (B, T) 24kHz waveform
        return: (B, 1, 31, 513) – StyleTTS2 JDC와 동일한 입력 형태에 가깝게 맞춤
        """
        device = wav.device

        # 1) power mel-spectrogram (B, 80, T')
        mel = self.mel_spec(wav)  # power
        mel_db = self.to_db(mel)  # log-mel (dB)

        # 2) StyleTTS2 JDC는 31 frame, 513 freq bin을 사용.
        #    여기서는 중앙 31 frame을 잘라 쓰고, freq 축은 zero-pad 또는 crop.
        B, n_mels, T = mel_db.shape  # (B, 80, T')
        if T < 31:
            pad_t = 31 - T
            mel_db = nn.functional.pad(mel_db, (0, pad_t))  # right pad in time
            T = 31

        # 중앙 31 frame 선택
        start = (T - 31) // 2
        mel_31 = mel_db[:, :, start:start + 31]  # (B, 80, 31)

        # freq axis를 513에 맞추기 위해 simple STFT-like 확장:
        # StyleTTS2 원코드는 mag-spec(513) 기반이지만, 여기서는 mel을
        # 간단히 linear projection 해서 513 차원으로 매핑.
        mel_31 = mel_31.transpose(1, 2)  # (B, 31, 80)
        W = torch.empty(80, 513, device=device)
        nn.init.xavier_uniform_(W)
        spec_31 = mel_31 @ W  # (B, 31, 513)
        spec_31 = spec_31.unsqueeze(1)  # (B, 1, 31, 513)

        return spec_31

    @torch.no_grad()
    def style_from_audio(self, wav):
        """
        wav : (B, T) waveform @ 24kHz
        return : (B, D) style vector
        """
        mel_jdc = self._wav_to_jdc_input(wav)  # (B, 1, 31, 513)
        cls_out, gan_feat, poolblock_out = self.ref_enc(mel_jdc)

        # gan_feat 또는 poolblock_out을 style feature로 사용.
        # 원 코드 기준 GAN_feature는 (B, 31, C, 2) 비슷한 형태라,
        # 모든 time/공간축 평균 → (B, C) 스타일 벡터
        dims = tuple(range(1, gan_feat.dim()))
        style = gan_feat.mean(dim=dims)  # (B, D)
        return style
