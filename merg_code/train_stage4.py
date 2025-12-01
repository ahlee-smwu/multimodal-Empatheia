
import torch, os
from config.cs_common import load_cs_config
from dataset import load_dataset
from model.cs_sd import ContentSynchronizer, StyleDisentangler
from model.styletts2_wrap import StyleTTS2Encoders
from model.dreamtalk_wrap import DreamTalkEncoders
from model.losses_cs_sd import loss_ccl, loss_sal, loss_cls
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

def main():
    cfg = load_cs_config('merg_code/config/cs_sd.yaml')
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    train_loader = load_dataset(mode='train', audio_path=None, video_path=None, batch_size=cfg.batch_size, num_workers=cfg.num_workers)

    tokenizer = AutoTokenizer.from_pretrained(cfg.llm_model_name, use_fast=True)
    llm = AutoModelForCausalLM.from_pretrained(cfg.llm_model_name, torch_dtype=torch.float16, device_map='auto', output_hidden_states=True)
    peft_cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout)
    llm = get_peft_model(llm, peft_cfg)

    cs = ContentSynchronizer(d_in=cfg.d_model, d_latent=cfg.d_latent_cs, d_out=cfg.d_model,
                             num_layers=cfg.num_layers, nhead=cfg.nhead, dim_ff=cfg.dim_ff).to(device)
    sd = StyleDisentangler(d_in=cfg.d_model, d_latent=cfg.d_latent_sd, d_out=cfg.d_model,
                           num_layers=cfg.num_layers, nhead=cfg.nhead, dim_ff=cfg.dim_ff).to(device)
    optim = torch.optim.AdamW(list(cs.parameters())+list(sd.parameters())+list(llm.parameters()), lr=cfg.lr, weight_decay=cfg.weight_decay)

    sty = StyleTTS2Encoders(cfg.styletts2_ckpt_dir).to(device)
    drm = DreamTalkEncoders(cfg.dreamtalk_ckpt_dir).to(device)

    step=0; cs.train(); sd.train(); llm.train()
    for batch in train_loader:
        dialogues = batch['conversations']
        targets   = batch['conversations'] if isinstance(dialogues[0], str) else [x['response_text'] for x in dialogues]
        inputs = [f"[DIALOGUE]\n{d}\n[TARGET]\n{t}" for d,t in zip(dialogues, targets)]
        tok = tokenizer(inputs, return_tensors='pt', padding=True, truncation=True, max_length=cfg.max_len).to(llm.device)
        out = llm(**tok, labels=tok['input_ids']); hs = out.hidden_states[-1]
        r_t, r_s, r_v = hs, hs, hs

        C_s, C_v, kld_cs = cs(r_t.to(device))
        S_s, S_v, logits, kld_sd = sd(r_s.to(device), r_v.to(device))

        wav = batch.get('audio', batch.get('wav', None)); video = batch.get('video', None)
        if wav is None or video is None: raise RuntimeError("DataLoader must provide 'audio' and 'video'.")
        C_s_gold = sty.text_content(targets).to(device)
        C_v_gold = drm.content_from_audio(wav.to(device))
        S_s_gold = sty.style_from_audio(wav.to(device))
        S_v_gold = drm.style_from_video(video.to(device))

        prof = batch['response_profile']
        labels = {
            'emotion': batch['response_emotion'].to(device),
            'age':     prof['age'].to(device),
            'gender':  prof['gender'].to(device),
            'tone':    (prof.get('timbre', None) or prof.get('tone')).to(device)
        }

        loss_emp = out.loss
        L = (loss_emp
             + cfg.alpha*loss_ccl(C_s, C_v, C_s_gold, C_v_gold)
             + cfg.beta*(loss_sal(S_s, S_v, S_s_gold, S_v_gold) + loss_cls(logits, labels))
             + cfg.kld_weight*(kld_cs + kld_sd))
        optim.zero_grad(set_to_none=True); L.backward(); optim.step()
        step+=1
        if step % cfg.log_every==0: print(f"[S4] step {step} L_total={L.item():.4f}  L_emp={loss_emp.item():.4f}")
        if step % cfg.save_every==0:
            import torch
            torch.save(cs.state_dict(), os.path.join(cfg.out_dir, 'cs.pt'))
            torch.save(sd.state_dict(), os.path.join(cfg.out_dir, 'sd.pt'))
            llm.save_pretrained(os.path.join(cfg.out_dir, 'lora'))
        if step>=cfg.max_steps_s4: break

    import torch
    torch.save(cs.state_dict(), os.path.join(cfg.out_dir, 'cs.pt'))
    torch.save(sd.state_dict(), os.path.join(cfg.out_dir, 'sd.pt'))
    llm.save_pretrained(os.path.join(cfg.out_dir, 'lora'))

if __name__=='__main__': main()
