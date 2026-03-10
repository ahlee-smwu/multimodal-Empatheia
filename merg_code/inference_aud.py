import os

# ------------------ distributed env ------------------
os.environ["LOCAL_RANK"] = "0"
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from header import *
from model import *
import torch
import torchaudio
import argparse
import logging
from dataset import load_dataset
from model import load_model
from config import load_config
from config.cs_common import load_cs_config
from model.cs_sd import ContentSynchronizer, StyleDisentangler
from model.styletts2_wrap import StyleTTS2Decoders
import glob
import torch.distributed as dist
import torch.nn.functional as F

logging.getLogger().setLevel(logging.ERROR)

# ------------------------------------------------
# Args
# ------------------------------------------------
def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='merg')
    parser.add_argument('--ckpt_path', type=str, default="ckpt/merg_ckpt/10000") # merg model ckpt
    parser.add_argument('--ckpt_module', type=str,
                        default='ckpt/merg-total_ckpt/20260224_130904/30000_30ep')
    parser.add_argument('--ckpt_aud', type=str,
                        default='ckpt/pretrained_ckpt/styletts2_ckpt/decoders')
    # Number of tokens to resample C_s to before feeding the TTS decoder.
    # CS outputs a fixed 15 tokens; resampling to a larger value gives longer
    # audio proportional to n_tokens (50 tokens ≈ ~2-3 s for short responses).
    # No gold response is needed — set higher for longer expected responses.
    parser.add_argument('--n_tokens', type=int, default=50,
                        help='Target C_s token count fed to decoder (determines audio length)')
    parser.add_argument('--out_dir', type=str, default='output')
    parser.add_argument('--mode', type=str, default='train') #train #test
    # parser.add_argument('--audio_path', type=str, default="/mnt/SSD_raid1/AvaMERG/audio_v5_0") # navi
    parser.add_argument('--audio_path', type=str, default="/mnt/HDD_raid1/AvaMERG_jhchoi/AvaMERG/audio_v5_0") # a6000
    # parser.add_argument('--video_path', type=str, default="/mnt/SSD_raid1/AvaMERG/video_v5_0") # navi
    parser.add_argument('--video_path', type=str, default="/mnt/HDD_raid1/AvaMERG_jhchoi/AvaMERG/video_v5_0") # a6000
    parser.add_argument('--video_path_frame', type=str, default="/mnt/HDD_raid1/AvaMERG_jhchoi/AvaMERG/video_frame") # a6000 preprocessed
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--max_length', type=int, default=1024)
    return parser.parse_args()


# ------------------------------------------------
# Inference
# ------------------------------------------------
@torch.no_grad()
def main(args):
    cfg_dict = load_config(vars(args))
    for k, v in cfg_dict.items():
        setattr(args, k, v)

    device = torch.device("cuda")

    # ------------------ dist ------------------
    if not dist.is_initialized():
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29502"
        dist.init_process_group(
            backend="nccl",
            init_method="env://"
        )

    # ------------------ config ------------------
    cfg = load_cs_config('merg_code/config/cs_sd.yaml')
    args.ds_config_path = 'merg_code/dsconfig/dsconfig_infer.json'
    args.dschf = HfDeepSpeedConfig(args.ds_config_path)
    args.total_steps = 1000  # dummy

    # ------------------ dataset ------------------
    train_data, train_iter, _ = load_dataset(vars(args))
    print(len(train_data))

    # ------------------ MERG agent (GPU + DeepSpeed) ------------------
    agent = load_model(vars(args))
    agent.ds_engine.eval()

    # ------------------ CS / SD (GPU) ------------------
    cs = ContentSynchronizer(
        d_in=cfg.d_in,
        d_latent=cfg.d_latent_cs,
        d_out=cfg.d_out,
        d_out_s=cfg.d_out_cs, # 512
        num_layers=cfg.num_layers,
        nhead=cfg.nhead,
        dim_ff=cfg.dim_ff,
    ).to(device).eval()

    sd = StyleDisentangler(
        d_in=cfg.d_in,
        d_latent=cfg.d_latent_sd,
        d_out=cfg.d_out_sd,
        d_out_s=cfg.d_out_ss,  # 128
        d_out_v=cfg.d_out_sv,  # 1024
        num_layers=cfg.num_layers,
        nhead=cfg.nhead,
        dim_ff=cfg.dim_ff,
    ).to(device).eval()

    cs_ckpt = glob.glob(os.path.join(args.ckpt_module, 'cs_*.pt'))[0]
    sd_ckpt = glob.glob(os.path.join(args.ckpt_module, 'sd_*.pt'))[0]

    cs.load_state_dict(torch.load(cs_ckpt, map_location=device))
    sd.load_state_dict(torch.load(sd_ckpt, map_location=device))

    print(f"✅ Loaded CS: {os.path.basename(cs_ckpt)}")
    print(f"✅ Loaded SD: {os.path.basename(sd_ckpt)}")

    # ------------------ StyleTTS2 (GPU) ------------------
    '''
    [ Content + Style + F0 + N ]
            ↓
    Acoustic Decoder (mel 생성)
            ↓
    HiFi-GAN Generator
            ↓
          waveform
    '''

    sty = StyleTTS2Decoders(args.ckpt_aud, device=device).eval()

    pbar = tqdm(
        train_iter,
        desc=f"iter",
        dynamic_ncols=True,
    )

    for batch in pbar:
        if batch is None:
            continue
        # ------------------ MERG forward (GPU) ------------------
        with torch.no_grad():
            outputs, *_ = agent.return_output(batch)
            hs = outputs.hidden_states[-1].float()

        # ------------------ CS / SD ------------------
        C_s, _, _ = cs(hs)
        S_s, _, _, _ = sd(hs, hs)
        # C_s: (B, 512, 15)  S_s: (B, 128)

        # ------------------------------------------------------------------
        # Resample C_s to --n_tokens length before feeding the decoder.
        #
        # CS always outputs a fixed 15 tokens (T_text in q_s_c).  The TTS
        # prosody predictor assigns a duration to each token and generates
        # that many frames, so 15 tokens → ~0.7 s of audio.
        #
        # During training, C_s was aligned to the gold phoneme count
        # (typically 15-25 tokens) via adaptive_avg_pool1d before loss was
        # computed.  At inference time no gold response is needed: we simply
        # resample to --n_tokens (default 50) so the decoder produces enough
        # frames for a full empathetic response (~2-3 s).  Increase
        # --n_tokens for longer expected responses.
        # ------------------------------------------------------------------
        if C_s.shape[-1] != args.n_tokens:
            C_s = F.adaptive_avg_pool1d(C_s, output_size=args.n_tokens)

        # ------------------ TTS ------------------
        wav = sty(C_s, S_s)

        # ------------------ save ------------------
        dia_id = batch['dia_ids'][0]  # batch size 1이면 이렇게
        out_dir = os.path.join(args.out_dir, 'audio')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f'out_{dia_id}.wav')
        x = wav[0].detach().cpu().squeeze()
        torchaudio.save(out_path, x.unsqueeze(0), 24000)

        print(f"✅ Saved: {out_path}")
        # print(f"C_s: {C_s.shape}, S_s: {S_s.shape}")


if __name__ == "__main__":
    args = parser_args()
    main(args)
