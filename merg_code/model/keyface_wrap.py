import sys
import os
import importlib.util


def force_import_audio_wrapper():
    possible_paths = [
        "/home/yjcho/multimodal-Empatheia/keyface_cvpr/scripts/util/audio_wrapper.py",
        "/home/yjcho/multimodal-Empatheia/scripts/util/audio_wrapper.py",
    ]

    for path in possible_paths:
        if os.path.exists(path):
            spec = importlib.util.spec_from_file_location("AudioWrapper", path)
            module = importlib.util.module_from_spec(spec)
            sys.modules["scripts.util.audio_wrapper"] = module
            spec.loader.exec_module(module)
            return module.AudioWrapper

    raise ImportError("audio_wrapper.py를 찾을 수 없습니다. 경로를 확인해주세요.")


try:
    AudioWrapper = force_import_audio_wrapper()
    print("\AudioWrapper forced import successful!")
except Exception as e:
    print(f"Error: {e}")
    raise e

import torch
import torch.nn as nn
from scripts.util.audio_wrapper import AudioWrapper


class KeyFaceEncoders(nn.Module):
    def __init__(self, keyface_ckpt_path, d_out=512):
        super().__init__()

        self.audio_encoder = AudioWrapper(model_type="wav2vec2")
        # self.audio_proj = nn.Linear(768, d_out)

        self.style_extractor = self._load_style_extractor(keyface_ckpt_path)
        # self.style_proj = nn.Linear(1024, d_out)

        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    def _load_style_extractor(self, ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = checkpoint["module"]

        weight_key = "_forward_module.conditioner.embedders.3.linear.weight"
        bias_key = "_forward_module.conditioner.embedders.3.linear.bias"

        extractor = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(3, 1536),
            nn.ReLU(),
            nn.Linear(1536, 1024)
        )

        with torch.no_grad():
            if weight_key in state_dict:
                extractor[4].weight.copy_(state_dict[weight_key])
                extractor[4].bias.copy_(state_dict[bias_key])
            else:
                print("No weight")

        return extractor

    @torch.no_grad()
    def content_from_audio(self, wav_batch):
        if wav_batch.dim() == 1:
            wav_batch = wav_batch.unsqueeze(0)  # (1, T)
        emb = self.audio_encoder.wav2vec2_encoding(wav_batch)
        if emb.dim() == 2:
            emb = emb.unsqueeze(0)  # (1, T', 768)
        # utterance-level로 쓸 거면
        if emb.dim() == 3:
            emb = emb.mean(dim=1)  # (B, 768)
        return emb

    @torch.no_grad()
    def style_from_video(self, vid_batch):
        ref_frame = vid_batch[:, 0]
        style_emb = self.style_extractor(ref_frame)
        return style_emb