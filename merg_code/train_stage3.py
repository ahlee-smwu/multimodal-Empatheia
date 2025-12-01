
import torch, os
from config.cs_common import load_cs_config
from dataset import load_dataset
from model.cs_sd import StyleDisentangler
from model.styletts2_wrap import StyleTTS2Encoders
from model.dreamtalk_wrap import DreamTalkEncoders
from model.losses_cs_sd import loss_sal, loss_cls
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    cfg = load_cs_config('merg_code/config/cs_sd.yaml')
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    train_loader = load_dataset(mode='train', audio_path=None, video_path=None, batch_size=cfg.batch_size, num_workers=cfg.num_workers)

    tokenizer = AutoTokenizer.from_pretrained(cfg.llm_model_name, use_fast=True)
    llm = AutoModelForCausalLM.from_pretrained(cfg.llm_model_name, torch_dtype=torch.float16, device_map='auto', output_hidden_states=True)
    for p in llm.parameters(): p.requires_grad_(False)

    sd = StyleDisentangler(d_in=cfg.d_model, d_latent=cfg.d_latent_sd, d_out=cfg.d_model,
                           num_layers=cfg.num_layers, nhead=cfg.nhead, dim_ff=cfg.dim_ff).to(device)
    optim = torch.optim.AdamW(sd.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    sty = StyleTTS2Encoders(cfg.styletts2_ckpt_dir).to(device)
    drm = DreamTalkEncoders(cfg.dreamtalk_ckpt_dir).to(device)

    step=0; sd.train()
    for batch in train_loader:
        dialogues = batch['conversations']
        targets   = batch['conversations'] if isinstance(dialogues[0], str) else [x['response_text'] for x in dialogues]
        inputs = [f"[DIALOGUE]\n{d}\n[TARGET]\n{t}" for d,t in zip(dialogues, targets)]
        tok = tokenizer(inputs, return_tensors='pt', padding=True, truncation=True, max_length=cfg.max_len).to(llm.device)
        out = llm(**tok, labels=tok['input_ids']); hs = out.hidden_states[-1]
        r_s, r_v = hs, hs  # if you segment by markers, split here

        wav = batch.get('audio', batch.get('wav', None)); video = batch.get('video', None)
        if wav is None or video is None:
            raise RuntimeError("Your DataLoader must supply 'audio' (wav) and 'video' tensors for SD alignment.")
        S_s_gold = sty.style_from_audio(wav.to(device)); S_v_gold = drm.style_from_video(video.to(device))
        S_s, S_v, logits, kld = sd(r_s.to(device), r_v.to(device))

        # labels from your batch (adjust keys to your dataset exactly)
        prof = batch['response_profile']
        labels = {
            'emotion': batch['response_emotion'].to(device),
            'age':     prof['age'].to(device),
            'gender':  prof['gender'].to(device),
            'tone':    (prof.get('timbre', None) or prof.get('tone')).to(device)
        }
        L = loss_sal(S_s, S_v, S_s_gold, S_v_gold) + loss_cls(logits, labels) + cfg.kld_weight * kld
        optim.zero_grad(set_to_none=True); L.backward(); optim.step()
        step+=1
        if step % cfg.log_every==0: print(f"[S3] step {step} L_total={L.item():.4f}")
        if step % cfg.save_every==0: torch.save(sd.state_dict(), os.path.join(cfg.out_dir, 'sd.pt'))
        if step>=cfg.max_steps_s3: break
    torch.save(sd.state_dict(), os.path.join(cfg.out_dir, 'sd.pt'))

if __name__=='__main__': main()
