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


def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--model', type=str, default='merg')
    parser.add_argument('--mode', type=str, default='train', help='train or test')
    parser.add_argument('--data_path', type=str, default='merg_data')
    parser.add_argument('--audio_path', type=str, default="/mnt/dataset/AvaMERG_jhchoi/AvaMERG/audio_v5_0")
    parser.add_argument('--video_path', type=str, default="/mnt/dataset/AvaMERG_jhchoi/AvaMERG/video_v5_0")
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

def main(**args):
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

    train_data, train_iter, sampler = load_dataset(args)
    train_num = train_data.__len__()
    print(f'################################# Num of training data #######################################: {train_num}')
    total_steps = args['epochs'] * train_num // dschf.config['train_batch_size']
    args['total_steps'] = total_steps

    agent = load_model(args)
    torch.distributed.barrier()

    tokenizer = AutoTokenizer.from_pretrained(cfg.llm_model_name, use_fast=True)
    llm = AutoModelForCausalLM.from_pretrained(cfg.llm_model_name, torch_dtype=torch.float16, device_map='auto', output_hidden_states=True)
    peft_cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout)
    llm = get_peft_model(llm, peft_cfg)

    cs = ContentSynchronizer(d_in=cfg.d_in, d_latent=cfg.d_latent_cs, d_out=cfg.d_out,
                             num_layers=cfg.num_layers, nhead=cfg.nhead, dim_ff=cfg.dim_ff).to(device)
    sd = StyleDisentangler(d_in=cfg.d_in, d_latent=cfg.d_latent_sd, d_out=cfg.d_out,
                           num_layers=cfg.num_layers, nhead=cfg.nhead, dim_ff=cfg.dim_ff).to(device)
    optim = torch.optim.AdamW(list(cs.parameters())+list(sd.parameters())+list(llm.parameters()), lr=cfg.lr, weight_decay=cfg.weight_decay)

    sty = StyleTTS2Encoders(cfg.styletts2_ckpt_dir).to(device)
    drm = sty
    # drm = DreamTalkEncoders(cfg.dreamtalk_ckpt_dir).to(device)

    step=0
    cs.train(); sd.train()
    # llm.train()

    for batch in train_iter:
        '''mllm(AvaMERG) Model'''
        outputs, inputs_embeds, input_ids, target_ids, attention_mask = agent.return_output(batch)

        # dialogues = batch['conversations']
        # targets = batch['conversations'] if isinstance(dialogues[0], str) else [x['response'] for x in dialogues]
        # inputs = [f"[DIALOGUE]\n{d}\n[TARGET]\n{t}" for d,t in zip(dialogues, targets)]
        # tok = tokenizer(inputs, return_tensors='pt', padding=True, truncation=True, max_length=cfg.max_len).to(llm.device)
        # out = llm(**tok, labels=tok['input_ids'])
        # hs = out.hidden_states[-1]

        hs = outputs.hidden_states[-1]
        r_t, r_s, r_v = hs, hs, hs

        '''CS/SD Modules'''
        C_s, C_v, kld_cs = cs(r_t.to(device))
        S_s, S_v, logits, kld_sd = sd(r_s.to(device), r_v.to(device))

        torch.save(r_t, "merg_code/model/cs_sd_tensor/r_t.pt")
        torch.save(r_s, "merg_code/model/cs_sd_tensor/r_s.pt")
        torch.save(r_v, "merg_code/model/cs_sd_tensor/r_v.pt")
        torch.save(C_s, "merg_code/model/cs_sd_tensor/C_s.pt")
        torch.save(C_v, "merg_code/model/cs_sd_tensor/C_v.pt")
        torch.save(kld_cs, "merg_code/model/cs_sd_tensor/kld_cs.pt")
        torch.save(S_s, "merg_code/model/cs_sd_tensor/S_s.pt")
        torch.save(S_v, "merg_code/model/cs_sd_tensor/S_v.pt")
        torch.save(logits, "merg_code/model/cs_sd_tensor/logits.pt")
        torch.save(kld_sd, "merg_code/model/cs_sd_tensor/kld_sd.pt")


        '''Generators encoding'''
        # 데이터셋의 audio/video를 바로 넣어야 generator에 encoding 해야 함
        wav = batch.get('audio', batch.get('wav', None));
        video = batch.get('video', None)
        if wav is None or video is None:
            raise RuntimeError("DataLoader must provide 'audio' and 'video'.")
        C_s_gold = sty.text_content(input_ids=input_ids, attention_mask=attention_mask).to(device)  # (B, proj_dim)
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

        '''save ckpt'''
        now = datetime.datetime.now()
        date_str = now.strftime("%Y%m%d_%H%M%S")  # 예: 20251202_131800
        ckpt_dir = os.path.join(cfg.out_dir, date_str)
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"Output directory: {ckpt_dir}")
        if step % cfg.save_every==0:
            torch.save(cs.state_dict(), os.path.join(ckpt_dir, f'cs_{step}-step.pt'))
            torch.save(sd.state_dict(), os.path.join(ckpt_dir, f'sd_{step}-step.pt'))
            llm.save_pretrained(os.path.join(ckpt_dir, f'lora_{step}-step'))
        if step>=cfg.max_steps_s4: break

    torch.save(cs.state_dict(), os.path.join(ckpt_dir, f'cs_{step}-step.pt'))
    torch.save(sd.state_dict(), os.path.join(ckpt_dir, f'sd_{step}-step.pt'))
    llm.save_pretrained(os.path.join(cfg.out_dir, f'lora_{step}-step'))

if __name__=='__main__':
    args = parser_args()
    args = vars(args)
    main(**args)
