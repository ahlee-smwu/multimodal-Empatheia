import os
import torch
from torch.utils.data import DataLoader

from config.cs_common import load_cs_config
from config import load_config
from dataset.all_dataset import multimodal_empathetic_dialogue
from model.cs_sd import StyleDisentangler
from model.styletts2_wrap import StyleTTS2Encoders
from model.dreamtalk_wrap import DreamTalkEncoders
from model.losses_cs_sd import loss_sal, loss_cls
from transformers import AutoTokenizer, AutoModelForCausalLM


def build_dataset(cfg_cs):
    base_cfg = load_config({})  # -> dict with ["models"]["data_path"], etc.

    ds_args = {
        "models": base_cfg["models"],
        "mode": "train",
    }

    dataset = multimodal_empathetic_dialogue(ds_args)
    loader = DataLoader(
        dataset,
        batch_size=cfg_cs.batch_size,
        shuffle=True,
        num_workers=cfg_cs.num_workers,
        collate_fn=dataset.collate_fn,
        pin_memory=True,
    )
    return dataset, loader


def build_inputs(dialogues):
    """Turn batch['conversations'] into text prompts for the LLM."""
    inputs = []
    for conv in dialogues:
        hist = conv.get("dialogue_history", [])

        # ---- dialogue history -> string ----
        if isinstance(hist, list):
            if len(hist) > 0 and isinstance(hist[0], dict):
                pieces = []
                for turn in hist:
                    spk = turn.get("speaker") or turn.get("role") or ""
                    utt = (
                        turn.get("utterance")
                        or turn.get("text")
                        or str(turn)
                    )
                    if spk:
                        pieces.append(f"{spk}: {utt}")
                    else:
                        pieces.append(str(utt))
                hist_txt = " ".join(pieces)
            else:
                hist_txt = " ".join(str(x) for x in hist)
        else:
            hist_txt = str(hist)

        # ---- gold response text ----
        resp = conv.get("response", conv.get("response_text", ""))
        if isinstance(resp, dict):
            resp = resp.get("text") or resp.get("utterance") or str(resp)

        txt = f"[DIALOGUE]\n{hist_txt}\n[TARGET]\n{resp}"
        inputs.append(txt)

    return inputs


def build_labels(batch, dataset, device):
    """Map metadata to integer class labels for loss_cls."""

    # emotion: string -> 7-class id (project via ED mapping)
    emo_ids = []
    for e in batch["response_emotion"]:
        base = dataset.ed_emotion_projection.get(e, e)
        emo_ids.append(dataset.emotion_projection.get(base, 0))

    emo_ids = torch.tensor(emo_ids, device=device, dtype=torch.long)

    # age / gender / timbre already numeric in __getitem__
    age_ids    = torch.tensor(batch["response_age"],    device=device, dtype=torch.long)
    gender_ids = torch.tensor(batch["response_gender"], device=device, dtype=torch.long)
    tone_ids   = torch.tensor(batch["response_timbre"], device=device, dtype=torch.long)

    # safety clamp
    n_emotions = len(dataset.emotion_projection)   # 7
    n_ages     = len(dataset.age_projection)       # 4
    n_genders  = len(dataset.gender_projection)    # 2
    n_timbres  = len(dataset.timbre_projection)    # 3

    emo_ids    = emo_ids.clamp(0, n_emotions - 1)
    age_ids    = age_ids.clamp(0, n_ages - 1)
    gender_ids = gender_ids.clamp(0, n_genders - 1)
    tone_ids   = tone_ids.clamp(0, n_timbres - 1)

    labels = {
        "emotion": emo_ids,
        "age":     age_ids,
        "gender":  gender_ids,
        "tone":    tone_ids,
    }
    return labels


def main():
    # ==================== config / devices ====================
    cfg = load_cs_config(
        "/home/a6000/bk-project/multimodal-Empatheia/merg_code/config/cs_sd.yaml"
    )
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    train_dataset, train_loader = build_dataset(cfg)

    # ==================== LLM backbone (Vicuna) ====================
    tokenizer = AutoTokenizer.from_pretrained(cfg.llm_model_name, use_fast=True)
    llm = AutoModelForCausalLM.from_pretrained(
        cfg.llm_model_name,
        torch_dtype=torch.float16,
        output_hidden_states=True,
    ).to(device)
    for p in llm.parameters():
        p.requires_grad_(False)
    llm.eval()

    # ==================== Style Disentangler (trainable) ====================
    sd = StyleDisentangler(
        d_in=cfg.d_in,
        d_latent=cfg.d_latent_sd,
        d_out=cfg.d_out,
        num_layers=cfg.num_layers,
        nhead=cfg.nhead,
        dim_ff=cfg.dim_ff,
    ).to(device)
    sd = sd.float()  # SD works in fp32
    optim = torch.optim.AdamW(sd.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # ==================== Gold style encoders (frozen) ====================
    # StyleTTS2: speech style (JDCNet)
    sty = StyleTTS2Encoders(cfg.styletts2_ckpt_dir).to(device)
    # DreamTalk replacement: video style (ResNet-18) + optional wav2vec2 content
    drm = DreamTalkEncoders(cfg.dreamtalk_ckpt_dir, d_out=cfg.d_out).to(device)

    sty.eval()
    drm.eval()
    for p in sty.parameters():
        p.requires_grad_(False)
    for p in drm.parameters():
        p.requires_grad_(False)

    # ==================== Training loop ====================
    step = 0
    sd.train()
    os.makedirs(cfg.out_dir, exist_ok=True)

    for batch in train_loader:
        # ---------- 1) text -> LLM hidden states ----------
        dialogues = batch["conversations"]
        inputs = build_inputs(dialogues)

        tok = tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg.max_len,
        )
        tok = {k: v.to(device) for k, v in tok.items() if k != "token_type_ids"}

        with torch.no_grad():
            out = llm(**tok)   # no labels; we only want hidden states

        hs = out.hidden_states[-1].to(device=device, dtype=torch.float32)  # [B,T,4096]
        r_s, r_v = hs, hs  # SD sees text-only representations for both branches

        B = hs.size(0)

        # ---------- 2) GOLD styles from real speech + video ----------
        # These *must* be provided by your Dataset.collate_fn as lists
        if "response_wav" not in batch or "response_video" not in batch:
            raise RuntimeError(
                "Stage3 SAL expects batch['response_wav'] and batch['response_video'] "
                "from the dataset collate_fn."
            )

        wav_list   = batch["response_wav"]    # list of length B, each Tensor or None
        video_list = batch["response_video"]  # list of length B, each Tensor or None

        gold_speech_styles = []
        gold_video_styles  = []

        with torch.no_grad():
            # ---- speech style via StyleTTS2 (JDCNet) ----
            for wav in wav_list:
                if wav is None:
                    gold_speech_styles.append(torch.zeros(cfg.d_out, device=device))
                    continue

                # expected shape (T,) or (1, T)
                if wav.dim() == 1:
                    wav_in = wav.unsqueeze(0)  # [1, T]
                elif wav.dim() == 2:
                    wav_in = wav
                else:
                    raise ValueError(f"Unexpected wav shape: {wav.shape}")

                wav_in = wav_in.to(device)
                s_vec = sty.style_from_audio(wav_in)      # [1, d_out]
                gold_speech_styles.append(s_vec.squeeze(0))

            # ---- video style via DreamTalkEncoders (ResNet-18) ----
            for vid in video_list:
                if vid is None:
                    gold_video_styles.append(torch.zeros(cfg.d_out, device=device))
                    continue

                # allowed shapes:
                # [C,H,W]      -> single frame
                # [T,C,H,W]    -> sequence
                # [B',C,H,W] or [B',T,C,H,W] won't happen here, we handle per-sample
                if vid.dim() == 3:        # [C,H,W]
                    v_in = vid.unsqueeze(0)          # [1,C,H,W]
                elif vid.dim() == 4:      # [T,C,H,W]
                    v_in = vid.unsqueeze(0)          # [1,T,C,H,W]
                elif vid.dim() == 5:      # already [1,T,C,H,W] maybe
                    v_in = vid
                else:
                    raise ValueError(f"Unexpected video shape: {vid.shape}")

                v_in = v_in.to(device)
                v_vec = drm.style_from_video(v_in)   # [1, d_out]
                gold_video_styles.append(v_vec.squeeze(0))

        S_s_gold = torch.stack(gold_speech_styles, dim=0)  # [B, d_out]
        S_v_gold = torch.stack(gold_video_styles,  dim=0)  # [B, d_out]

        # ---------- 3) SD forward (predict styles from text) ----------
        S_s, S_v, logits, kld = sd(r_s, r_v)  # S_* : [B, d_out]

        labels = build_labels(batch, train_dataset, device)

        # ---------- 4) Loss: Style Alignment + classification + KL ----------
        sal = loss_sal(S_s, S_v, S_s_gold, S_v_gold)
        cls = loss_cls(logits, labels)
        loss = sal + cls + cfg.kld_weight * kld

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        step += 1
        if step % cfg.log_every == 0:
            print(f"[Stage3] step {step}  L_total={loss.item():.4f}")

        if step % cfg.save_every == 0:
            torch.save(sd.state_dict(), os.path.join(cfg.out_dir, "sd.pt"))

        if step >= cfg.max_steps_s3:
            break

    torch.save(sd.state_dict(), os.path.join(cfg.out_dir, "sd.pt"))


if __name__ == "__main__":
    main()
