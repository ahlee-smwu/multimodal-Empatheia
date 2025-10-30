import os.path
import types
import json
from collections import OrderedDict
import torch
from torch.utils.tensorboard import SummaryWriter
import deepspeed
import datetime
import logging


class DeepSpeedAgent:

    def __init__(self, model, args):
        super(DeepSpeedAgent, self).__init__()
        self.args = args
        self.model = model

        self.print_model_parameters()
        self.writer = SummaryWriter(args['log_path'])

        if self.args['mode'] == 'test':
            self.load_parameters(self.args['save_path'], self.args['epochs'])

        # load config parameters of deepspeed
        ds_params = json.load(open(self.args['ds_config_path']))
        ds_params['scheduler']['params']['total_num_steps'] = self.args['total_steps']
        ds_params['scheduler']['params']['warmup_num_steps'] = max(10, int(
            self.args['total_steps'] * self.args['warmup_rate']))

        # ✅ optimizer를 PyTorch AdamW로 변경
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # DeepSpeed 초기화 시 optimizer 전달
        self.ds_engine, _, _, _ = deepspeed.initialize(
            model=self.model,
            model_parameters=self.model.parameters(),
            config_params=ds_params,
            optimizer=optimizer,
            dist_init_required=True,
            args=types.SimpleNamespace(**args)
        )

    @torch.no_grad()
    def predict(self):
        self.ds_engine.module.eval()
        output = self.ds_engine.generate(self.args)
        return output

    def train_model(self, batch, current_step=0, pbar=None):
        self.ds_engine.module.train()
        loss_dict = self.ds_engine(batch)
        for k, v in loss_dict.items():
            self.writer.add_scalar(k, v, current_step)

        loss = loss_dict['loss']
        if 'gen_acc' in loss_dict.keys():
            mle_acc = loss_dict['gen_acc']
        else:
            mle_acc = 0
        self.ds_engine.backward(loss)
        self.ds_engine.step()
        pbar.set_description(f'[!] loss: {round(loss.item(), 4)}; token_acc: {round(mle_acc * 100, 2)}')
        pbar.update(1)
        if self.args['local_rank'] == 0 and self.args['log_path'] and current_step % self.args['logging_step'] == 0:
            elapsed = pbar.format_dict['elapsed']
            rate = pbar.format_dict['rate']
            remaining = (pbar.total - pbar.n) / rate if rate and pbar.total else 0
            remaining = str(datetime.timedelta(seconds=remaining))
            logging.info(
                f'[!] progress: {round(pbar.n / pbar.total, 5)}; remaining time: {remaining}; loss: {round(loss.item(), 4)}; token_acc: {round(mle_acc * 100, 2)}')

        mle_acc *= 100
        return mle_acc

    def save_model(self, path, epoch, current_step):
        path = os.path.join(path, f'{epoch}')
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        checkpoint = OrderedDict()
        for k, v in self.ds_engine.module.named_parameters():
            if v.requires_grad or 'llama_proj' in k:
                checkpoint[k] = v
        torch.save(checkpoint, f'{path}/pytorch_model.pt')

        # save tokenizer & config
        self.model.llama_tokenizer.save_pretrained(path)
        self.model.llama_model.config.save_pretrained(path)
        print(f'[!] save model into {path}')

    def print_model_parameters(self, use_4bit=False):
        trainable_params = 0
        all_param = 0
        lora = 0
        ccl = 0
        sdm = 0
        linear = 0
        llama = 0
        imagebind = 0
        for name, param in self.model.named_parameters():
            num_params = param.numel()
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            if 'lora' in name:
                lora += num_params
            elif 'llama_proj' in name:
                linear += num_params
            elif 'llama_model' in name:
                llama += num_params
            elif 'visual_encoder' in name:
                imagebind += num_params

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        if use_4bit:
            trainable_params /= 2
        print(
            f"all params: {all_param:,d} || trainable params: {trainable_params:,d} || trainable%: {100 * trainable_params / all_param}"
        )
        print(f'lora params: {lora:,d} || ccl params: {ccl:,d} || sdm params: {sdm:,d}')
        print(f'linear params: {linear:,d} || imagebind params: {imagebind:,d} || llama params: {llama:,d}')

    def load_parameters(self, path):
        if os.path.exists(os.path.join(path, 'pytorch_model.pt')):
            print('#########################################################')
            print(f'loading parameters from {path}')
            print('#########################################################')
            delta_ckpt = torch.load(f'{path}/pytorch_model.pt', map_location=torch.device('cuda'))
            self.model.load_state_dict(delta_ckpt, strict=False)
