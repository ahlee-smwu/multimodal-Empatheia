from header import *
from dataset import load_dataset
from model import *
from config import load_config
import torch
import datetime, os
from config.cs_common import load_cs_config
from model.cs_sd import ContentSynchronizer, StyleDisentangler
from model.styletts2_wrap import StyleTTS2Encoders
from model.keyface_wrap import KeyFaceEncoders
from model.losses_cs_sd import loss_ccl, loss_sal, loss_cls
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
import torchaudio
import decord
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
logging.getLogger().setLevel(logging.ERROR)


def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--model', type=str, default='merg')
    parser.add_argument('--mode', type=str, default='train', help='train or test')
    parser.add_argument('--data_path', type=str, default='merg_data')
    # parser.add_argument('--audio_path', type=str, default="/home/elicer/bk/dataset/audio_v5_0") # elice
    # parser.add_argument('--audio_path', type=str, default="/mnt/SSD_raid1/AvaMERG/audio_v5_0") # navi
    parser.add_argument('--audio_path', type=str, default="/mnt/HDD_raid1/AvaMERG_jhchoi/AvaMERG/audio_v5_0") # a6000
    # parser.add_argument('--video_path', type=str, default="/home/elicer/bk/dataset/video_v5_0") # elice
    # parser.add_argument('--video_path', type=str, default="/mnt/SSD_raid1/AvaMERG/video_v5_0") # navi
    parser.add_argument('--video_path', type=str, default="/mnt/HDD_raid1/AvaMERG_jhchoi/AvaMERG/video_v5_0") # a6000
    parser.add_argument('--ckpt_path', type=str, default="ckpt/merg_ckpt/10000")
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--save_path', type=str, default='ckpt/merg_ckpt_total/')
    parser.add_argument('--log_path', type=str, default='ckpt/merg_ckpt_total/')
    parser.add_argument('--assets_path', type=str, default='./assets/')
    parser.add_argument('--max_length', type=int, default=1024)
    parser.add_argument('--cs_path', type=str, default='ckpt/merg-total_ckpt/20260214_182100/5000/cs_5000.pt')
    parser.add_argument('--sd_path', type=str, default='ckpt/merg-total_ckpt/20260214_182100/5000/sd_5000.pt')
    parser.add_argument('--styletts2_ckpt_dir', type=str, default='ckpt/pretrained_ckpt/styletts2_ckpt')
    parser.add_argument('--keyface_ckpt_dir', type=str, default='ckpt/pretrained_ckpt/keyface_ckpt')
    return parser.parse_args()


def initialize_distributed(args):
    # args: argparse.Namespace 를 직접 수정
    args.master_ip = os.getenv('MASTER_ADDR', 'localhost')
    args.master_port = os.getenv('MASTER_PORT', '6000')
    args.world_size = int(os.getenv('WORLD_SIZE', '1'))
    # RANK 환경 변수가 있으면 우선, 없으면 기존 local_rank 유지
    rank = int(os.getenv('RANK', str(args.local_rank)))
    args.local_rank = rank % torch.cuda.device_count()
    device = args.local_rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    deepspeed.init_distributed(dist_backend='nccl')


def load_wav_batch(path_list, device):
    wavs = []
    for p in path_list:
        if p is None:
            continue
        try:
            w, sr = torchaudio.load(p)
        except Exception as e:
            print(f"[WARN] failed to load wav {p}: {e}")
            return None
        if w.dim() == 2:
            w = w.mean(0)
        wavs.append(w)

    if len(wavs) == 0:
        return None

    wavs = torch.nn.utils.rnn.pad_sequence(wavs, batch_first=True)
    return wavs.to(device)


def load_video_batch(path_list, device, num_frames=8):
    vids = []
    for p in path_list:
        if p is None:
            continue
        try:
            vr = decord.VideoReader(p)
        except Exception as e:
            print(f"[WARN] failed to load video {p}: {e}")
            return None
        if len(vr) == 0:
            print(f"[WARN] empty video {p}")
            return None

        idx = np.linspace(0, len(vr) - 1, num_frames).astype(int)
        batch = vr.get_batch(idx)

        if isinstance(batch, torch.Tensor):
            f = batch.detach().cpu()
        else:
            f = torch.from_numpy(batch.asnumpy())

        f = f.permute(0, 3, 1, 2).float()
        vids.append(f)

    if len(vids) == 0:
        return None

    T_max = max(v.shape[0] for v in vids)
    B = len(vids)
    C, H, W = vids[0].shape[1:]
    out = torch.zeros(B, T_max, C, H, W)
    for i, v in enumerate(vids):
        T = v.shape[0]
        out[i, :T] = v

    return out.to(device)

def align_cs_to_gold(C_s, C_s_gold):
    """
    Align CS output temporal length to StyleTTS2 TextEncoder output
    Args:
        C_s:      (B, 768, T_cs)
        C_s_gold: (B, 768, T_text)
    Returns:
        C_s_aligned: (B, 768, T_text)
    """
    T_target = C_s_gold.size(-1)

    if C_s.size(-1) == T_target:
        return C_s

    return F.interpolate(
        C_s,
        size=T_target,
        mode="linear",
        align_corners=False
    )

def main(args):
    # load_config 가 dict 를 받는 경우를 유지
    cfg_dict = load_config(vars(args))
    for k, v in cfg_dict.items():
        setattr(args, k, v)

    args.ds_config_path = 'merg_code/dsconfig/dsconfig.json'
    dschf = HfDeepSpeedConfig(args.ds_config_path)
    args.dschf = dschf

    initialize_distributed(args)
    cfg = load_cs_config('merg_code/config/cs_sd.yaml')

    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(cfg.out_dir, exist_ok=True)

    train_data, train_iter, sampler = load_dataset(vars(args))
    train_num = len(train_data)

    total_steps = (args.epochs * train_num // dschf.config['train_batch_size'])
    args.total_steps = total_steps

    agent = load_model(vars(args))
    torch.distributed.barrier()

    cs = ContentSynchronizer(
        d_in=cfg.d_in,
        d_latent=cfg.d_latent_cs,
        d_out=cfg.d_out,
        num_layers=cfg.num_layers,
        nhead=cfg.nhead,
        dim_ff=cfg.dim_ff,
    ).to(device)

    sd = StyleDisentangler(
        d_in=cfg.d_in,
        d_latent=cfg.d_latent_sd,
        d_out=cfg.d_out,
        num_layers=cfg.num_layers,
        nhead=cfg.nhead,
        dim_ff=cfg.dim_ff,
    ).to(device)

    # ✅ is_main_process를 먼저 정의
    is_main_process = (args.local_rank == 0)

    if args.cs_path and args.sd_path:
        cs.load_state_dict(torch.load(args.cs_path, map_location=device))
        sd.load_state_dict(torch.load(args.sd_path, map_location=device))
        if is_main_process:
            print(f"✅ Loaded CS from {args.cs_path}")
            print(f"✅ Loaded SD from {args.sd_path}")
    else:
        if is_main_process:
            print(f"❌ Checkpoint not found: {args.cs_path}, {args.sd_path}")

    optim = torch.optim.AdamW(
        list(cs.parameters()) + list(sd.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    sty = StyleTTS2Encoders(os.path.join(args.styletts2_ckpt_dir, 'encoders')).to(device)
    keyface = KeyFaceEncoders(os.path.join(args.keyface_ckpt_dir, 'keyframe.pt'), d_out=cfg.d_out).to(device)

    for m in [sty, keyface]:
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)

    step = 0
    agent.ds_engine.eval()
    agent.ds_engine.requires_grad_(False)
    cs.train()
    sd.train()

    now = datetime.datetime.now()
    base_dir = os.path.join("ckpt/merg-total_ckpt", now.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(base_dir, exist_ok=True)

    writer = SummaryWriter(base_dir) if is_main_process else None

    def normalize_label(x, device):
        if isinstance(x, torch.Tensor):
            return x.to(device).view(-1).long()
        if isinstance(x, list):
            return torch.tensor(x, device=device, dtype=torch.long).view(-1)
        return torch.tensor([x], device=device, dtype=torch.long())

    for epoch in range(args.epochs):
        pbar = tqdm(
            train_iter,
            desc=f"Epoch {epoch+1}/{args.epochs}",
            dynamic_ncols=True,
            disable=not is_main_process,
        )

        for batch in pbar:
            if batch is None:
                continue
            with torch.no_grad():
                outputs, _, _, _, _ = agent.return_output(batch)
                hs = outputs.hidden_states[-1].float()
                loss_emp = outputs.loss.detach().float()

            r_t, r_s, r_v = hs, hs, hs

            C_s, C_v, kld_cs = cs(r_t.to(device))
            S_s, S_v, logits, kld_sd = sd(r_s.to(device), r_v.to(device))

            responses = [conv["response"] for conv in batch["conversations"]]
            C_s_gold = sty.text_content(responses).to(device).float()

            wav_batch = batch['response_audio'].to(device, non_blocking=True)
            vid_batch = batch['response_video'].to(device, non_blocking=True)

            C_v_gold = keyface.content_from_audio(wav_batch).float()
            S_s_gold = sty.style_from_audio(wav_batch).reshape(-1, 192).to(device).float()
            S_v_gold = keyface.style_from_video(vid_batch).float()

            labels = {
                'emotion': normalize_label(batch['response_emotion'], device),
                'age': normalize_label(batch['response_age'], device),
                'gender': normalize_label(batch['response_gender'], device),
                'tone': normalize_label(batch['response_timbre'], device),
            }

            '''
            C_s: torch.Size([1, 768])
            C_s g: torch.Size([1, 768, 18])
            C_v: torch.Size([1, 768])
            C_v g: torch.Size([1, 768])
            '''
            C_s_aligned = align_cs_to_gold(C_s, C_s_gold)

            L_ccl = loss_ccl(C_s_aligned, C_v, C_s_gold, C_v_gold)
            L_sal = loss_sal(S_s, S_v, S_s_gold, S_v_gold)
            L_cls = loss_cls(logits, labels)

            L_train = (
                cfg.alpha * L_ccl
                + cfg.beta * (L_sal + L_cls)
                + cfg.kld_weight * (kld_cs + kld_sd)
            )

            optim.zero_grad(set_to_none=True)
            L_train.backward()
            optim.step()
            step += 1

            if is_main_process:
                pbar.set_postfix(
                    step=step,
                    L_ccl=f"{L_ccl.item():.3f}",
                    L_sal_cls=f"{(L_sal + L_cls).item():.3f}",
                    KLD=f"{(kld_cs + kld_sd).item():.3f}",
                    L_total=f"{(loss_emp + L_train).item():.3f}",
                )

                writer.add_scalar("loss/L_ccl", L_ccl.item(), step)
                writer.add_scalar("loss/L_sal_cls", (L_sal + L_cls).item(), step)
                writer.add_scalar("loss/KLD", (kld_cs + kld_sd).item(), step)
                writer.add_scalar("loss/L_total", (loss_emp + L_train).item(), step)

                if step % cfg.save_every == 0 and step > 0:
                    ckpt_dir = os.path.join(base_dir, f"{step}")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    torch.save(cs.state_dict(), os.path.join(ckpt_dir, f'cs_{step}.pt'))
                    torch.save(sd.state_dict(), os.path.join(ckpt_dir, f'sd_{step}.pt'))

            if step >= cfg.max_steps_s4:
                break

        if step >= cfg.max_steps_s4:
            break

    if is_main_process and writer is not None:
        writer.close()


if __name__ == '__main__':
    args = parser_args()
    main(args)
