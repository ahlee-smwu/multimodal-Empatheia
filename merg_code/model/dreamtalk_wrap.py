
import torch, torch.nn as nn, os

class DreamTalkEncoders(nn.Module):
    def __init__(self, ckpt_dir):
        super().__init__()
        self.audio_enc = torch.jit.load(os.path.join(ckpt_dir, 'audio_encoder.pt'))
        self.style_enc = torch.jit.load(os.path.join(ckpt_dir, 'style_encoder.pt'))
        for m in [self.audio_enc, self.style_enc]:
            m.eval().requires_grad_(False)

    @torch.no_grad()
    def content_from_audio(self, wav):
        return self.audio_enc(wav)

    @torch.no_grad()
    def style_from_video(self, video_tensor):
        return self.style_enc(video_tensor)
