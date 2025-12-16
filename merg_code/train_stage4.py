from header import *
from dataset import load_dataset
from model import *
from config import load_config
import torch
import datetime, os
from config.cs_common import load_cs_config
from dataset import load_dataset
from model.cs_sd import ContentSynchronizer, StyleDisentangler
from model.styletts2_wrap import StyleTTS2Encoders
from model.dreamtalk_wrap import DreamTalkEncoders
from model.losses_cs_sd import loss_ccl, loss_sal, loss_cls
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
import torchaudio
import decord
import numpy as np



def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--model', type=str, default='merg')
    parser.add_argument('--mode', type=str, default='train', help='train or test')
    parser.add_argument('--data_path', type=str, default='merg_data')
    parser.add_argument('--audio_path', type=str, default="/mnt/dataset/AvaMERG_jhchoi/AvaMERG/audio_v5_0")
    parser.add_argument('--video_path', type=str, default="/mnt/dataset/AvaMERG_jhchoi/AvaMERG/video_v5_0")
    parser.add_argument('--ckpt_path', type=str, default="ckpt/merg_ckpt/10000")
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--save_path', type=str, default='ckpt/merg_ckpt_total/')
    parser.add_argument('--log_path', type=str, default='ckpt/merg_ckpt_total/')
    parser.add_argument('--assets_path', type=str, default='./assets/')
    parser.add_argument('--max_length', type=int, default=1024)

    return parser.parse_args()

def initialize_distributed(args):
    args['master_ip'] = os.getenv('MASTER_ADDR', 'localhost')
    args['master_port'] = os.getenv('MASTER_PORT', '6000')
    args['world_size'] = int(os.getenv('WORLD_SIZE', '1'))
    args['local_rank'] = int(os.getenv('RANK', '0')) % torch.cuda.device_count()
    device = args['local_rank'] % torch.cuda.device_count()
    torch.cuda.set_device(device)
    deepspeed.init_distributed(dist_backend='nccl')
def load_wav_batch(path_list, device):
    """
    path_list: list of str (wav paths)
    returns: [B, T_max] float32 mono, or None if something fails
    """
    wavs = []
    for p in path_list:
        if p is None:
            continue
        try:
            w, sr = torchaudio.load(p)  # [C, T]
        except Exception as e:
            print(f"[WARN] failed to load wav {p}: {e}")
            return None
        if w.dim() == 2:      # [C, T] -> mono
            w = w.mean(0)     # [T]
        wavs.append(w)

    if len(wavs) == 0:
        return None

    wavs = torch.nn.utils.rnn.pad_sequence(wavs, batch_first=True)  # [B, T_max]
    return wavs.to(device)

def load_video_batch(path_list, device, num_frames=8):
    """
    path_list: list of str (video paths)
    returns: [B, T, C, H, W] float32 in [0,1], or None on failure
    """
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
        batch = vr.get_batch(idx)  # could be decord NDArray or torch.Tensor

        # --- FIX HERE ---
        if isinstance(batch, torch.Tensor):
            # decord torch bridge: already a tensor [T, H, W, C]
            f = batch.detach().cpu()
        else:
            # standard decord NDArray: use asnumpy()
            f = torch.from_numpy(batch.asnumpy())
        # ----------------

        # [T, H, W, C] -> [T, C, H, W], float32
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

def main(**args):
    '''config'''
    args = load_config(args)
    args['ds_config_path'] = f'merg_code/dsconfig/dsconfig.json'
    dschf = HfDeepSpeedConfig(args['ds_config_path'])
    args['dschf'] = dschf
    print(args)
    initialize_distributed(args)
    cfg = load_cs_config('merg_code/config/cs_sd.yaml')
    print(cfg)
    import torch
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(cfg.out_dir, exist_ok=True)

    '''dataset'''
    train_data, train_iter, sampler = load_dataset(args)
    train_num = train_data.__len__()
    print(f'################################# Num of training data #######################################: {train_num}')
    total_steps = args['epochs'] * train_num // dschf.config['train_batch_size']
    args['total_steps'] = total_steps

    '''MLLM(AvaMERG) model'''
    agent = load_model(args)
    torch.distributed.barrier()
    '''CS/CD module'''
    cs = ContentSynchronizer(d_in=cfg.d_in, d_latent=cfg.d_latent_cs, d_out=cfg.d_out,
                             num_layers=cfg.num_layers, nhead=cfg.nhead, dim_ff=cfg.dim_ff).to(device)
    sd = StyleDisentangler(d_in=cfg.d_in, d_latent=cfg.d_latent_sd, d_out=cfg.d_out,
                           num_layers=cfg.num_layers, nhead=cfg.nhead, dim_ff=cfg.dim_ff).to(device)
    optim = torch.optim.AdamW(
        list(cs.parameters()) +
        list(sd.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    '''Generator encoder module'''
    sty = StyleTTS2Encoders(cfg.styletts2_ckpt_dir).to(device)
    drm = DreamTalkEncoders(cfg.dreamtalk_ckpt_dir, d_out=cfg.d_out).to(device)
    # drm = DreamTalkEncoders(cfg.dreamtalk_ckpt_dir).to(device)
    for m in [sty, drm]:
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
    # ---------- training loop ----------
    step = 0
    agent.ds_engine.eval()  # LLM frozen
    cs.train()
    sd.train()

    # we’ll reuse one base ckpt_dir for the whole run
    now = datetime.datetime.now()
    base_dir = os.path.join(cfg.out_dir, now.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(base_dir, exist_ok=True)
    print(f"Base output directory: {base_dir}")

    def normalize_label(x, device):
        if isinstance(x, torch.Tensor):
            x = x.to(device)
            x = x.view(-1)
            return x.long()
        if isinstance(x, list):
            return torch.tensor(x, device=device, dtype=torch.long).view(-1)
        return torch.tensor([x], device=device, dtype=torch.long)
    for batch in train_iter:
        # ---------------- MLLM forward (FROZEN) ----------------
        with torch.no_grad():
            outputs, inputs_embeds, input_ids, target_ids, attention_mask = agent.return_output(batch)
            hs = outputs.hidden_states[-1].float()  # [B, T, d_in] in fp32
            loss_emp = outputs.loss.detach().float()  # scalar fp32, no grad

        r_t, r_s, r_v = hs, hs, hs

        # ---------------- CS / SD ----------------
        C_s, C_v, kld_cs = cs(r_t.to(device))
        S_s, S_v, logits, kld_sd = sd(r_s.to(device), r_v.to(device))

        # ensure KLDs are floats
        kld_cs = kld_cs.float()
        kld_sd = kld_sd.float()

        # ---------------- Gold encoders ----------------
        responses = [conv["response"] for conv in batch["conversations"]]
        C_s_gold = sty.text_content(responses).to(device).float()  # (B, d_out)

        response_aud_paths = [p for sub in batch["response_audio"] for p in sub]
        response_vid_paths = [p for sub in batch["response_video"] for p in sub]

        if (response_aud_paths is None or len(response_aud_paths) == 0 or
                response_vid_paths is None or len(response_vid_paths) == 0):
            continue

        wav_batch = load_wav_batch(response_aud_paths, device)  # [B, T]
        vid_batch = load_video_batch(response_vid_paths, device)  # [B, T, C, H, W]

        if wav_batch is None or vid_batch is None:
            continue

        C_v_gold = drm.content_from_audio(wav_batch).float()  # (B, d_out)
        S_s_gold = sty.style_from_audio(response_aud_paths).reshape(-1, 192).to(device).float()
        S_v_gold = drm.style_from_video(vid_batch).float()  # (B, d_out)

        # ---------------- labels for L_cls ----------------
        labels = {
            'emotion': normalize_label(batch['response_emotion'], device),
            'age': normalize_label(batch['response_age'], device),
            'gender': normalize_label(batch['response_gender'], device),
            'tone': normalize_label(batch['response_timbre'], device),
        }

        # ---------------- individual loss terms (in fp32) ----------------
        L_ccl = loss_ccl(C_s, C_v, C_s_gold, C_v_gold).float()
        L_sal = loss_sal(S_s, S_v, S_s_gold, S_v_gold).float()
        L_cls = loss_cls(logits, labels).float()

        # loss used for *training* (only cs/sd / gold paths)
        L_train = (
                cfg.alpha * L_ccl
                + cfg.beta * (L_sal + L_cls)
                + cfg.kld_weight * (kld_cs + kld_sd)
        )

        # loss used for logging: include empathy loss
        L_total = loss_emp + L_train

        optim.zero_grad(set_to_none=True)
        L_train.backward()  # <--- ONLY backprop through your modules
        optim.step()
        step += 1

        if step % cfg.log_every == 0:
            print(f"[S4] step {step}  L_total={L_total.item():.4f}  "
                  f"L_emp={loss_emp.item():.4f}")

        if step % cfg.save_every == 0:
            ckpt_dir = os.path.join(base_dir, f"step_{step}")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(cs.state_dict(), os.path.join(ckpt_dir, f'cs_{step}-step.pt'))
            torch.save(sd.state_dict(), os.path.join(ckpt_dir, f'sd_{step}-step.pt'))

        if step >= cfg.max_steps_s4:
            break

        # final save
    ckpt_dir = os.path.join(base_dir, f"final_{step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(cs.state_dict(), os.path.join(ckpt_dir, f'cs_{step}-step.pt'))
    torch.save(sd.state_dict(), os.path.join(ckpt_dir, f'sd_{step}-step.pt'))


if __name__ == '__main__':
    args = parser_args()
    args = vars(args)
    main(**args)


