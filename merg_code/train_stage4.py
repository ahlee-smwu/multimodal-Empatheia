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
    drm = sty
    # drm = DreamTalkEncoders(cfg.dreamtalk_ckpt_dir).to(device)

    '''train loop'''
    step=0
    agent.ds_engine.eval() # frozen
    cs.train(); sd.train()

    for batch in train_iter:
        ''' # batch content
        {'dia_ids': ['17677'],
         'conversations': [{'dialogue_history': [{'index': 0,
                                                  'role': 'speaker',
                                                  'utterance': "I just feel like I'm constantly juggling everything and it's wearing me down."},
                                                 {'index': 1,
                                                  'role': 'listener',
                                                  'utterance': 'That sounds really tough. It must be exhausting to manage so much all at once.'},
                                                 {'index': 2,
                                                  'role': 'speaker',
                                                  'utterance': "It's like no matter how hard I try, I can't seem to find a balance."},
                                                 {'index': 3,
                                                  'role': 'listener',
                                                  'utterance': 'Finding that balance can be really challenging, especially with so many expectations.'},
                                                 {'index': 4,
                                                  'role': 'speaker',
                                                  'utterance': "I just wish I could catch a break and feel like I'm on top of things again."}],
                            'response': "Everyone has moments like these, and it's okay to ask for help when you need it.",
                            'coe': {'speaker_emotion': 'anxious',
                                    'event_scenario': 'Feeling the need for a break',
                                    'emotion_cause': 'The pressure of managing responsibilities without relief',
                                    'goal_to_response': 'To find reassurance that seeking help is acceptable'}}],
         'response_age': [2],
         'response_emotion': [4],
         'response_gender': [0],
         'response_timbre': [2],
         'response_profile': [14],
         'response_audio': [/mnt~ path],
         'response_video': [/mnt~ path]} '''

        '''MLLM(AvaMERG) model'''
        outputs, inputs_embeds, input_ids, target_ids, attention_mask = agent.return_output(batch)

        hs = outputs.hidden_states[-1]
        r_t, r_s, r_v = hs, hs, hs

        '''CS/SD Modules'''
        C_s, C_v, kld_cs = cs(r_t.to(device))  # (B,768)
        S_s, S_v, logits, kld_sd = sd(r_s.to(device), r_v.to(device))  # (B,192), (B,768)

        # torch.save(r_t, "merg_code/model/cs_sd_tensor/r_t.pt")
        # torch.save(r_s, "merg_code/model/cs_sd_tensor/r_s.pt")
        # torch.save(r_v, "merg_code/model/cs_sd_tensor/r_v.pt")
        # torch.save(C_s, "merg_code/model/cs_sd_tensor/C_s.pt")
        # torch.save(C_v, "merg_code/model/cs_sd_tensor/C_v.pt")
        # torch.save(kld_cs, "merg_code/model/cs_sd_tensor/kld_cs.pt")
        # torch.save(S_s, "merg_code/model/cs_sd_tensor/S_s.pt")
        # torch.save(S_v, "merg_code/model/cs_sd_tensor/S_v.pt")
        # torch.save(logits, "merg_code/model/cs_sd_tensor/logits.pt")
        # torch.save(kld_sd, "merg_code/model/cs_sd_tensor/kld_sd.pt")

        '''Generator encoder module'''
        # 데이터셋의 audio/video를 바로 넣어야 generator에 encoding 해야 함

        response = [conv["response"] for conv in batch["conversations"]]
        response_aud = [item for sublist in batch["response_audio"] for item in sublist]
        response_vid = [item for sublist in batch["response_video"] for item in sublist]
        if (response_vid is None or len(response_vid) == 0 or
            response_aud is None or len(response_aud) == 0
        ):
            continue

        C_s_gold = sty.text_content(response).to(device)  # (B, 768)
        C_v_gold = torch.zeros(1, 768).to(device) #drm.content_from_audio(response_aud).to(device)
        S_s_gold = sty.style_from_audio(response_aud).reshape(-1, 192).to(device) # (B,192)
        S_v_gold = torch.zeros(1, 768).to(device) #drm.style_from_video(response_vid).to(device)

        def normalize_label(x, device):
            if isinstance(x, torch.Tensor):
                x = x.to(device)
                x = x.view(-1)
                return x.long()
            if isinstance(x, list):
                return torch.tensor(x, device=device, dtype=torch.long).view(-1)
            return torch.tensor([x], device=device, dtype=torch.long)

        labels = {
            'emotion': normalize_label(batch['response_emotion'], device),
            'age': normalize_label(batch['response_age'], device),
            'gender': normalize_label(batch['response_gender'], device),
            'tone': normalize_label(batch['response_timbre'], device)
            # 'profile': batch['response_profile'] # summurize age/gender/timbre
            # emotion/timbre를 묶어서 집중 처리하면 좋을 듯
        }

        # TO-DO: loss dtype matching
        L = (cfg.alpha*loss_ccl(C_s, C_v, C_s_gold, C_v_gold)
             + cfg.beta*(loss_sal(S_s, S_v, S_s_gold, S_v_gold) + loss_cls(logits, labels))
             + cfg.kld_weight*(kld_cs + kld_sd))
        optim.zero_grad(set_to_none=True);
        L.backward();
        optim.step()
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
            # llm.save_pretrained(os.path.join(ckpt_dir, f'lora_{step}-step'))
        if step>=cfg.max_steps_s4: break

    torch.save(cs.state_dict(), os.path.join(ckpt_dir, f'cs_{step}-step.pt'))
    torch.save(sd.state_dict(), os.path.join(ckpt_dir, f'sd_{step}-step.pt'))
    # llm.save_pretrained(os.path.join(cfg.out_dir, f'lora_{step}-step'))

if __name__=='__main__':
    args = parser_args()
    args = vars(args)
    main(**args)
