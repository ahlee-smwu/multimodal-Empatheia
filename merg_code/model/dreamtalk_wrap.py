# merg_code/model/dreamtalk_wrap.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model
import torchvision.models as models


class DreamTalkEncoders(nn.Module):
    """
    Replacement for the original DreamTalk encoders.

    - Audio encoder  : wav2vec2-base (HF)
    - Video encoder  : ResNet-18 (ImageNet)
    - Both are frozen and projected to d_out (e.g., 768) so they match the SD module.
    """

    def __init__(
        self,
        ckpt_dir: str = None,        # kept for compatibility, not used
        d_out: int = 768,
        audio_model_name: str = "facebook/wav2vec2-base-960h",
    ):
        super().__init__()

        # ---- audio encoder: wav2vec2 ----
        self.audio_model = Wav2Vec2Model.from_pretrained(audio_model_name)
        self.audio_hidden = self.audio_model.config.hidden_size
        self.audio_proj = nn.Linear(self.audio_hidden, d_out)

        # ---- video encoder: ResNet-18 ----
        self.video_backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        )
        # remove final FC, get 512-dim features
        self.video_backbone.fc = nn.Identity()
        self.video_feat_dim = 512
        self.video_proj = nn.Linear(self.video_feat_dim, d_out)

        # freeze backbones
        for p in self.audio_model.parameters():
            p.requires_grad_(False)
        for p in self.video_backbone.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def content_from_audio(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Approximate 'content' embedding from audio.

        wav: [B, T] or [B, 1, T] waveform (16kHz is ideal)
        returns: [B, d_out]
        """
        if wav.dim() == 3 and wav.size(1) == 1:
            wav = wav[:, 0, :]
        elif wav.dim() != 2:
            raise ValueError(f"Unexpected wav shape: {wav.shape}")

        wav = wav.float().to(self.audio_proj.weight.device)

        out = self.audio_model(wav)
        # [B, T', H] -> mean over time
        feat = out.last_hidden_state.mean(dim=1)   # [B, H]
        feat = self.audio_proj(feat)               # [B, d_out]
        return feat

    @torch.no_grad()
    def style_from_video(self, video_tensor: torch.Tensor) -> torch.Tensor:
        """
        Approximate 'style' embedding from video.

        video_tensor:
          - [B, T, C, H, W]  or
          - [B, C, T, H, W]  or
          - [B, C, H, W]
        returns: [B, d_out]
        """
        v = video_tensor

        if v.dim() == 5:
            # either [B, T, C, H, W] or [B, C, T, H, W]
            if v.size(2) == 3:  # [B, C, T, H, W]
                B, C, T, H, W = v.shape
                v = v.permute(0, 2, 1, 3, 4)  # -> [B, T, C, H, W]
            else:  # [B, T, C, H, W]
                B, T, C, H, W = v.shape

            v = v.reshape(B * T, C, H, W)  # [B*T, C, H, W]
        elif v.dim() == 4:
            # [B, C, H, W]
            B, C, H, W = v.shape
            T = 1
        else:
            raise ValueError(f"Unexpected video shape: {v.shape}")

        device = self.video_proj.weight.device
        v = v.float().to(device)

        # assume input in [0,1] or [0,255]
        if v.max() > 1.5:
            v = v / 255.0

        # resize to 224x224 for ResNet
        v = F.interpolate(v, size=(224, 224), mode="bilinear", align_corners=False)

        # ImageNet normalization
        mean = v.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = v.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        v = (v - mean) / std

        feat = self.video_backbone(v)          # [B*T, 512]
        feat = feat.view(B, T, -1).mean(1)     # [B, 512]
        feat = self.video_proj(feat)           # [B, d_out]
        return feat
