import os
import json
import datetime
from collections import OrderedDict
from header import *

import torch
from torch.utils.tensorboard import SummaryWriter

class DeepSpeedAgent:

    def __init__(self, model, args):
        super(DeepSpeedAgent, self).__init__()
        self.args = args
        self.model = model

        self.print_model_parameters()
        self.writer = SummaryWriter(args['log_path'])


        if self.args['mode']: # org: test mode
            # self.load_parameters(os.path.join(self.args['save_path'], str(self.args['epochs'])))
            self.load_parameters(os.path.join('ckpt/merg_ckpt/1'))

        # load config parameters of deepspeed
        ds_params = json.load(open(self.args['ds_config_path']))
        ds_params['scheduler']['params']['total_num_steps'] = self.args['total_steps']
        ds_params['scheduler']['params']['warmup_num_steps'] = max(10, int(
            self.args['total_steps'] * self.args['warmup_rate']))

        # ✅ method 1) use deepspeed optimizer
        # self.ds_engine, self.optimizer, _, _ = deepspeed.initialize(
        #     model=self.model,
        #     model_parameters=self.model.parameters(),
        #     config_params=ds_params,
        #     dist_init_required=True,
        #     args=types.SimpleNamespace(**args)
        # )

        # ✅ method 2) use normal pytorch optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
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
        print("ds_engine has generate?", hasattr(self.ds_engine, "generate"))
        print("wrapped module has generate?", hasattr(self.ds_engine.module, "generate"))
        output = self.ds_engine.generate(self.args)
        return output

    def train_model(self, batch, current_step=0, pbar=None):
        self.ds_engine.module.train()
        loss_dict = self.ds_engine(batch)
        for k,v in loss_dict.items():
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

    def return_output(self, batch):
        self.ds_engine.module.train()
        outputs, inputs_embeds, input_ids, target_ids, attention_mask = self.ds_engine.forward_llm(batch)

        return outputs, inputs_embeds, input_ids, target_ids, attention_mask

    def test_model(self, batch, current_step=0, pbar=None):
        self.ds_engine.module.eval()  # 평가 모드
        with torch.no_grad():
            # 모델 forward: DeepSpeed 엔진은 일반적으로 batch만 받으면 됨
            loss_dict = self.ds_engine(batch)
        for k, v in loss_dict.items():
            if hasattr(self, "writer"):
                self.writer.add_scalar(f"test/{k}", v, current_step)
        loss = loss_dict['loss']
        if 'gen_acc' in loss_dict:
            mle_acc = loss_dict['gen_acc']
        else:
            mle_acc = 0
        # tqdm 진행바 업데이트
        if pbar is not None:
            pbar.set_description(f"[Test] loss: {round(loss.item(), 4)}; token_acc: {round(mle_acc * 100, 2)}")
            pbar.update(1)
        return loss.item(), mle_acc * 100

    def inference_model(self, batch):
        self.ds_engine.module.eval()  # 평가 모드
        with torch.no_grad():
            # 모델 forward: DeepSpeed 엔진은 일반적으로 batch만 받으면 됨
            inputs_embeds, attention_mask = self.ds_engine.module.return_generate(batch)
            gen_ids = self.ds_engine.module.llama_model.generate(
                input_ids=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=128,
                temperature=0.8,
                top_p=0.95,
                do_sample=True
            )

        gen_text = self.model.llama_tokenizer.decode(gen_ids[0], skip_special_tokens=False)
        print("추론 결과:", gen_text)

        return inputs_embeds, attention_mask

    def save_model(self, path, epoch, current_step):
        """
            this function also save the trainable parameters and specific name parameters
        """
        path = os.path.join(path, f'{epoch}')

        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        param_grad_dic = {
            k: v.requires_grad for (k, v) in self.ds_engine.module.named_parameters()
        }
        state_dict = self.ds_engine.module.state_dict()
        checkpoint = OrderedDict()
        for k, v in self.ds_engine.module.named_parameters():
            if v.requires_grad:
                checkpoint[k] = v
            if 'llama_proj' in k:
                checkpoint[k] = v
        torch.save(checkpoint, f'{path}/pytorch_model.pt')
        # save tokenizer
        self.model.llama_tokenizer.save_pretrained(path)
        # save configuration
        self.model.llama_model.config.save_pretrained(path)
        print(f'[!] save model into {path}')

    def print_model_parameters(self, use_4bit=False):
        """
            Prints the number of trainable parameters in the model.
            """
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
            # if using DS Zero 3 and the weights are initialized empty
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
            else:
                pass

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
            print('loading parameters from {}'.format(path))
            print('#########################################################')
            delta_ckpt = torch.load(f'{path}/pytorch_model.pt', map_location=torch.device('cuda'))

            checkpoint = OrderedDict()
            checkpoint = delta_ckpt
            self.model.load_state_dict(checkpoint, strict=False)


class AccelerateAgent:
    def __init__(self, model, args):
        super(AccelerateAgent, self).__init__()
        self.args = args
        self.model = model  # should be a torch.nn.Module (unwrapped before prepare)
        self.optimizer = None  # main에서 설정하거나 기본 옵티 생성
        self.accelerator = None  # main에서 할당 (Accelerator 객체)
        self.model_device = None

        self.print_model_parameters()
        self.writer = SummaryWriter(args['log_path'])

        # optional: if user wants to load saved checkpoint at init
        if self.args.get('mode') == 'test':
            # user may provide path to directory containing pytorch_model.pt
            self.load_parameters(self.args['save_path'])

    def _forward_and_get_loss_dict(self, batch):
        """
        Try several common call styles:
        - model(batch)
        - model(**batch)  (if batch is dict)
        The model is expected to return either:
        - a dict that contains 'loss' and optionally other metrics
        - a tensor representing loss
        """
        self.model.train()
        out = None
        try:
            # try dict unpack first if batch is dict-like
            if isinstance(batch, dict):
                out = self.model(**batch)
            else:
                out = self.model(batch)
        except TypeError:
            # fallback: try passing the single batch element
            out = self.model(batch)

        if isinstance(out, dict):
            loss_dict = out
        elif torch.is_tensor(out):
            loss_dict = {'loss': out}
        else:
            # maybe tuple: (loss, others)
            if isinstance(out, (list, tuple)) and len(out) >= 1 and torch.is_tensor(out[0]):
                loss_dict = {'loss': out[0]}
            else:
                raise RuntimeError("Model forward returned unsupported type. Expected dict or tensor or tuple(starting with tensor).")
        return loss_dict

    def forward_loss(self, batch):
        """batch를 넣고 loss tensor 반환"""
        self.model.train()
        loss_dict = None
        try:
            if isinstance(batch, dict):
                out = self.model(**batch)
            else:
                out = self.model(batch)
        except TypeError:
            out = self.model(batch)

        if isinstance(out, dict):
            loss_dict = out
        elif torch.is_tensor(out):
            loss_dict = {'loss': out}
        elif isinstance(out, (list, tuple)) and len(out) > 0 and torch.is_tensor(out[0]):
            loss_dict = {'loss': out[0]}
        else:
            raise RuntimeError("Model forward returned unsupported type")

        loss = loss_dict.get('loss')
        if loss is None:
            raise RuntimeError("No 'loss' key found in model output")

        return loss, loss_dict

    def predict(self, **gen_kwargs):
        """
        generate / inference helper. If underlying model exposes .generate or .forward for generation.
        gen_kwargs are forwarded to model.generate if available.
        """
        self.model.eval()
        with torch.no_grad():
            # if model has generate (LLM), use it
            if hasattr(self.model, "generate"):
                return self.model.generate(**gen_kwargs)
            # else try a predict/generate method on model wrapper
            if hasattr(self.model, "predict"):
                return self.model.predict(**gen_kwargs)
            # otherwise, raise
            raise RuntimeError("Model has no generate/predict method. Provide generation entry or implement predict() in agent.")

    def train_model(self, batch, current_step=0, pbar=None):
        """
        Single training step using standard optimizer step and accelerator.backward if available.
        Assumes:
         - self.model is already moved/prepared (e.g., by accelerator.prepare)
         - self.optimizer exists and points to prepared optimizer
         - if using Accelerator for mixed precision, self.accelerator should be set
        """
        if self.model is None:
            raise RuntimeError("self.model is None. Did you forget to set agent.model from load_model / accelerator.prepare?")

        loss_dict = self._forward_and_get_loss_dict(batch)

        # log scalars
        for k, v in loss_dict.items():
            # v might be tensor or float
            try:
                val = v.item() if torch.is_tensor(v) else float(v)
            except Exception:
                val = float(v)
            self.writer.add_scalar(k, val, current_step)

        loss = loss_dict.get('loss')
        if loss is None:
            raise RuntimeError("No 'loss' key found in model output.")

        # backward (use accelerator if available)
        if hasattr(self, 'accelerator') and self.accelerator is not None:
            self.accelerator.backward(loss)
        else:
            loss.backward()

        # optimizer step
        if self.optimizer is None:
            raise RuntimeError("No optimizer set on agent. Set agent.optimizer in main (after accelerator.prepare).")
        self.optimizer.step()
        # zero grads
        self.optimizer.zero_grad()

        # optional metrics
        mle_acc = loss_dict.get('gen_acc', 0)
        try:
            mle_acc_val = mle_acc.item() if torch.is_tensor(mle_acc) else float(mle_acc)
        except Exception:
            mle_acc_val = 0.0

        # progress bar update/description (only if caller provided pbar)
        if pbar is not None:
            try:
                pbar.set_description(f'[!] loss: {round(loss.item(), 4)}; token_acc: {round(mle_acc_val * 100, 2)}')
                pbar.update(1)
            except Exception:
                pass

        # logging to file periodically (only from main process — check accelerator if available)
        is_main = True
        if hasattr(self, 'accelerator') and self.accelerator is not None:
            is_main = self.accelerator.is_main_process

        if is_main and self.args.get('log_path') and current_step % max(1, self.args.get('logging_step', 100)) == 0:
            if pbar is not None:
                elapsed = pbar.format_dict.get('elapsed', 0)
                rate = pbar.format_dict.get('rate', 0)
                remaining = (pbar.total - pbar.n) / rate if rate and pbar.total else 0
                remaining = str(datetime.timedelta(seconds=remaining))
            else:
                elapsed = 0
                remaining = "unknown"
            logging.info(
                f'[!] progress: {round(pbar.n / pbar.total, 5) if pbar else 0}; remaining time: {remaining}; loss: {round(loss.item(), 4)}; token_acc: {round(mle_acc_val * 100, 2)}')

        return mle_acc_val * 100

    def save_model(self, path, epoch, current_step):
        """
        Save model state_dict and tokenizer/config if present.
        Uses accelerator.unwrap_model if accelerator is set to get the real module.
        """
        outdir = os.path.join(path, f'{epoch}')
        if not os.path.exists(outdir):
            os.makedirs(outdir, exist_ok=True)

        # if wrapped by accelerator, unwrap
        save_obj = self.model
        if hasattr(self, 'accelerator') and self.accelerator is not None:
            try:
                save_obj = self.accelerator.unwrap_model(self.model)
            except Exception:
                save_obj = self.model

        # collect trainable parameters checkpoint (mimic previous behavior)
        checkpoint = OrderedDict()
        for k, v in save_obj.named_parameters():
            if v.requires_grad or 'llama_proj' in k:
                checkpoint[k] = v.detach().cpu()

        torch.save(checkpoint, os.path.join(outdir, 'pytorch_model.pt'))

        # save tokenizer / config if attributes exist
        if hasattr(save_obj, 'llama_tokenizer'):
            try:
                save_obj.llama_tokenizer.save_pretrained(outdir)
            except Exception:
                pass
        if hasattr(save_obj, 'llama_model') and hasattr(save_obj.llama_model, 'config'):
            try:
                save_obj.llama_model.config.save_pretrained(outdir)
            except Exception:
                pass

        print(f'[!] save model into {outdir}')

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
        ckpt_path = os.path.join(path, 'pytorch_model.pt')
        if os.path.exists(ckpt_path):
            print('loading parameters from', ckpt_path)
            delta_ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
            try:
                # attempt to load into model (may be partially matched)
                self.model.load_state_dict(delta_ckpt, strict=False)
                print("loaded state dict into model (strict=False).")
            except Exception as e:
                print("Error loading state dict with strict=False:", e)
        else:
            print("checkpoint not found at", ckpt_path)
