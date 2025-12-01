
import torch, os
from config.cs_common import load_cs_config
from dataset import load_dataset
from model.cs_sd import ContentSynchronizer
from model.styletts2_wrap import StyleTTS2Encoders
from model.dreamtalk_wrap import DreamTalkEncoders
from model.losses_cs_sd import loss_ccl
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

def main():
    cfg = load_cs_config('merg_code/config/cs_sd.yaml')
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    train_loader = load_dataset(mode='train', audio_path=None, video_path=None, batch_size=cfg.batch_size, num_workers=cfg.num_workers)

    # LLM frozen
    tokenizer = AutoTokenizer.from_pretrained(cfg.llm_model_name, use_fast=True)
    llm = AutoModelForCausalLM.from_pretrained(cfg.llm_model_name, torch_dtype=torch.float16, device_map='auto', output_hidden_states=True)
    for p in llm.parameters(): p.requires_grad_(False)

    cs = ContentSynchronizer(d_in=cfg.d_model, d_latent=cfg.d_latent_cs, d_out=cfg.d_model,
                             num_layers=cfg.num_layers, nhead=cfg.nhead, dim_ff=cfg.dim_ff).to(device)
    optim = torch.optim.AdamW(cs.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    sty = StyleTTS2Encoders(cfg.styletts2_ckpt_dir).to(device)
    drm = DreamTalkEncoders(cfg.dreamtalk_ckpt_dir).to(device)

    step=0; cs.train()
    for batch in train_loader:
        dialogues = batch['conversations']
        targets   = batch['conversations'] if isinstance(dialogues[0], str) else [x['response_text'] for x in dialogues]
        # forward LLM to get r_t tokens after [DIALOGUE]
        inputs = [f"[DIALOGUE]\n{d}\n[TARGET]\n{t}" for d,t in zip(dialogues, targets)]
        tok = tokenizer(inputs, return_tensors='pt', padding=True, truncation=True, max_length=cfg.max_len).to(llm.device)
        out = llm(**tok, labels=tok['input_ids'])
        hs = out.hidden_states[-1]
        r_t = hs  # treat whole sequence as r_t; optional: segment if your tokenizer has markers

        # golds
        C_s_gold = sty.text_content(targets).to(device)
        # need audio waveform batch for video-content alignment — expect loader to include 'audio' tensor as 'audio' or 'wav'
        wav = batch.get('audio', batch.get('wav', None))
        if wav is None:
            raise RuntimeError("Your DataLoader must supply gold response speech as 'audio' or 'wav'.")
        C_v_gold = drm.content_from_audio(wav.to(device))

        C_s, C_v, kld = cs(r_t.to(device))
        L = loss_ccl(C_s, C_v, C_s_gold, C_v_gold) + cfg.kld_weight * kld
        optim.zero_grad(set_to_none=True); L.backward(); optim.step()
        step+=1
        if step % cfg.log_every==0: print(f"[S2] step {step} L_ccl={L.item():.4f}")
        if step % cfg.save_every==0: torch.save(cs.state_dict(), os.path.join(cfg.out_dir, 'cs.pt'))
        if step>=cfg.max_steps_s2: break
    torch.save(cs.state_dict(), os.path.join(cfg.out_dir, 'cs.pt'))

if __name__=='__main__': main()
