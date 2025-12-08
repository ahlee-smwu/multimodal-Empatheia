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
    """
    Stage 3: build MERG text dataset without DeepSpeed / distributed.
    Uses the same JSON (merg_data/train.json) as Stage 1.
    """
    # load base.yaml (gives models.data_path etc.)
    base_cfg = load_config({})  # -> dict

    ds_args = {
        "models": base_cfg["models"],  # contains 'data_path'
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
        # conv is usually a dict with 'dialogue_history' and 'response'
        hist = conv.get("dialogue_history", [])

        # --- build history text ---
        if isinstance(hist, list):
            if len(hist) > 0 and isinstance(hist[0], dict):
                # e.g. [{'speaker': 'A', 'text': 'hi'}, ...]
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
                # list of plain strings / tokens
                hist_txt = " ".join(str(x) for x in hist)
        else:
            # already a single string or something else
            hist_txt = str(hist)

        # --- build response text ---
        resp = conv.get("response", conv.get("response_text", ""))

        if isinstance(resp, dict):
            # same pattern: pull out text field if present
            resp = resp.get("text") or resp.get("utterance") or str(resp)

        txt = f"[DIALOGUE]\n{hist_txt}\n[TARGET]\n{resp}"
        inputs.append(txt)

    return inputs


def build_labels(batch, dataset, device):
    """
    Map batch metadata to integer class labels for loss_cls.
    Uses the projections already defined inside multimodal_empathetic_dialogue.
    """

    # --- emotion (string) -> 7-class id ---
    emo_ids = []
    for e in batch["response_emotion"]:
        # normalize via ED projection, then map to 7-class
        base = dataset.ed_emotion_projection.get(e, e)
        emo_ids.append(dataset.emotion_projection.get(base, 0))

    emo_ids = torch.tensor(emo_ids, device=device, dtype=torch.long)

    # --- age / gender / timbre are already ints in __getitem__ ---
    age_ids    = torch.tensor(batch["response_age"],    device=device, dtype=torch.long)
    gender_ids = torch.tensor(batch["response_gender"], device=device, dtype=torch.long)
    tone_ids   = torch.tensor(batch["response_timbre"], device=device, dtype=torch.long)

    # ------- SAFETY: clamp all labels to valid ranges -------
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
    # 1) CS/SD hyperparams
    cfg = load_cs_config("/home/a6000/bk-project/multimodal-Empatheia/merg_code/config/cs_sd.yaml")
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # 2) dataset + dataloader (NO deepspeed, NO distributed)
    train_dataset, train_loader = build_dataset(cfg)

    # 3) LLM backbone (Vicuna)
    tokenizer = AutoTokenizer.from_pretrained(cfg.llm_model_name, use_fast=True)
    llm = AutoModelForCausalLM.from_pretrained(
        cfg.llm_model_name,
        torch_dtype=torch.float16,
        output_hidden_states=True,
    ).to(device)
    for p in llm.parameters():
        p.requires_grad_(False)
    llm.eval()  # add this line

    # 4) Style Disentangler
    sd = StyleDisentangler(
        d_in=cfg.d_in,              # 4096 (Vicuna hidden size)
        d_latent=cfg.d_latent_sd,   # 256
        d_out=cfg.d_out,            # 768
        num_layers=cfg.num_layers,
        nhead=cfg.nhead,
        dim_ff=cfg.dim_ff,
    ).to(device)
    sd = sd.float()
    optim = torch.optim.AdamW(sd.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # 5) Pretrained style encoders (hooked but not yet used in Stage 3 text-only training)
    sty = StyleTTS2Encoders(cfg.styletts2_ckpt_dir).to(device)
    drm = DreamTalkEncoders(cfg.dreamtalk_ckpt_dir, d_out=cfg.d_out).to(device)
    _ = (sty, drm)  # kept so loading is tested, not used here

    # 6) Training loop
    step = 0
    sd.train()
    os.makedirs(cfg.out_dir, exist_ok=True)

    for batch in train_loader:
        # (a) text ? LLM hidden states [B, T, 4096]
        dialogues = batch["conversations"]
        inputs = build_inputs(dialogues)

        tok = tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg.max_len,
        ).to(device)

        # Move tensors to device and DROP token_type_ids (not used by LLaMA/Vicuna)
        tok = {k: v.to(device) for k, v in tok.items() if k != "token_type_ids"}

        # --- no LM loss, just hidden states ---
        with torch.no_grad():
            out = llm(**tok)  # DO NOT pass labels

        hs = out.hidden_states[-1]  # [B, T, 4096]
        hs = hs.to(device=device, dtype=torch.float32)  # <-- cast to Float32 for SD
        r_s, r_v = hs, hs

        # (b) StyleDisentangler forward
        S_s, S_v, logits, kld = sd(r_s, r_v)

        # (c) "gold" style for SAL � text-only, so we just detach (SAL = 0 but shapes OK)
        S_s_gold, S_v_gold = S_s.detach(), S_v.detach()

        labels = build_labels(batch, train_dataset, device)

        loss = (
            loss_sal(S_s, S_v, S_s_gold, S_v_gold)  # this becomes 0 but keeps API consistent
            + loss_cls(logits, labels)
            + cfg.kld_weight * kld
        )

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
