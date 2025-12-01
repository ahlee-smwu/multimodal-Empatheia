
import torch, os
from config.cs_common import load_cs_config
from dataset import load_dataset
from model.common.modeling_llama import LlamaForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

def main():
    cfg = load_cs_config('merg_code/config/cs_sd.yaml')
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    train_loader = load_dataset(mode='train', audio_path=None, video_path=None, batch_size=cfg.batch_size, num_workers=cfg.num_workers)

    tokenizer = AutoTokenizer.from_pretrained(cfg.llm_model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(cfg.llm_model_name, torch_dtype=torch.float16, device_map='auto', output_hidden_states=True)
    peft_cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout)
    model = get_peft_model(model, peft_cfg)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    step=0; model.train()
    for batch in train_loader:
        # Build minimal prompt from your batch
        dialogues = batch['conversations']
        targets   = batch['conversations'] if isinstance(dialogues[0], str) else [x['response_text'] for x in dialogues]
        inputs = []
        for d, t in zip(dialogues, targets):
            seq = f"[DIALOGUE]\n{d}\n[TARGET]\n{t}"
            inputs.append(seq)
        tok = tokenizer(inputs, return_tensors='pt', padding=True, truncation=True, max_length=cfg.max_len).to(model.device)
        out = model(**tok, labels=tok['input_ids'])
        loss = out.loss
        optim.zero_grad(set_to_none=True); loss.backward(); optim.step()
        step+=1
        if step % cfg.log_every==0: print(f"[S1] step {step} loss_emp={loss.item():.4f}")
        if step % cfg.save_every==0: model.save_pretrained(os.path.join(cfg.out_dir, "lora"))
        if step>=cfg.max_steps_s1: break
    model.save_pretrained(os.path.join(cfg.out_dir, "lora"))

if __name__=='__main__': main()
