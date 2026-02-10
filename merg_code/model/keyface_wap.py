import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model
import torchvision.models as models

KEYFACE_DIR = "../keyface_cvpr"
if os.path.exists(KEYFACE_DIR):
    if KEYFACE_DIR not in sys.path:
        sys.path.append(KEYFACE_DIR)
        sys.path.append(os.path.join(KEYFACE_DIR, "sgm"))
    print(f"[INFO] KeyFace path found: {KEYFACE_DIR}")
else:

    print(f"[ERROR] KeyFace folder NOT FOUND at: {KEYFACE_DIR}")


class KeyFaceEncoders(nn.Module):
    def __init__(self, ckpt_path=None, d_out=768, audio_model_name="facebook/wav2vec2-base-960h"):
        super().__init__()

        self.audio_model = Wav2Vec2Model.from_pretrained(audio_model_name)
        self.audio_proj = nn.Linear(self.audio_model.config.hidden_size, d_out)

        self.video_backbone = models.resnet18(weights=None)
        self.video_backbone.fc = nn.Identity()
        self.video_proj = nn.Linear(512, d_out)
        if ckpt_path is None:
            ckpt_path = "../../ckpt/pretrained_ckpt/keyface_ckpt/keyframe.pt"

        if os.path.exists(ckpt_path):
            print(f"[INFO] Loading Head Generator Weights: {ckpt_path}")

            weights = torch.load(ckpt_path, map_location='cpu')

            self.video_backbone.load_state_dict(weights, strict=False)
        else:
            print(f"[WARNING] Checkpoint missing: {ckpt_path}")

        for p in self.audio_model.parameters(): p.requires_grad_(False)
        for p in self.video_backbone.parameters(): p.requires_grad_(False)

    @torch.no_grad()
    def content_from_audio(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.dim() == 3: wav = wav[:, 0, :]
        device = self.audio_proj.weight.device
        out = self.audio_model(wav.to(device).float())
        feat = out.last_hidden_state.mean(dim=1)
        return self.audio_proj(feat)

    @torch.no_grad()
    def style_from_video(self, video_tensor: torch.Tensor) -> torch.Tensor:
        v = video_tensor
        B, T, C, H, W = (v.shape if v.dim() == 5 else (v.size(0), 1, v.size(1), v.size(2), v.size(3)))
        v = v.reshape(B * T, C, H, W).float()
        device = self.video_proj.weight.device
        v = F.interpolate(v.to(device), size=(224, 224), mode="bilinear")
        if v.max() > 1.5: v = v / 255.0
        mean = v.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1);
        std = v.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        v = (v - mean) / std
        feat = self.video_backbone(v)
        return self.video_proj(feat.view(B, T, -1).mean(1))