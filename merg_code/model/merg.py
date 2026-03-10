import logging
import os.path
from typing import List
import torch
from header import *
import torch.nn.functional as F
from .ImageBind import *
from .ImageBind import data
from .common.modeling_llama import LlamaForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList
from .common.utils import *
from .cs_sd import ContentSynchronizer, StyleDisentangler
from .losses_cs_sd import loss_ccl, loss_sal, loss_cls
# from speech_generator.generate_audio import StyleTTS2
# from talking_face_generator.generate_video import generate_video
from .styletts2_wrap import StyleTTS2Encoders
from .keyface_wrap import KeyFaceEncoders
import soundfile as sf
import cv2
import glob

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops: List = None, encounters: int = 1):
        super().__init__()
        self.stops = [torch.tensor(stop, dtype=torch.long).cuda() for stop in stops]
        self.ENCOUNTERS = encounters

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop_token in self.stops:
            if input_ids.shape[1] >= len(stop_token): 
                if torch.equal(input_ids[0, -len(stop_token):], stop_token):
                    return True 
        return False

    
class MERGModel(nn.Module):
    """LoRA for LLaMa model"""

    def __init__(self, **args):
        super(MERGModel, self).__init__()
        self.args = args
        self.max_length = args['max_length']
        self.device = torch.cuda.current_device()
        print('args max_length', args['max_length'])

        self._init_language_model()
        self._init_imagebind()
        self.llama_tokenizer.add_tokens('<Vid>') 
        self.llama_tokenizer.add_tokens('<Aud>')  # add special token to tokenizer
        self.llama_model.resize_token_embeddings(len(self.llama_tokenizer))
        print('Tokenizer initialized.')
        self.input_embeddings = self.llama_model.get_input_embeddings()

        #encoding_multimodal
        self.llama_proj = nn.Linear(
            self.visual_hidden_size, self.llama_model.config.hidden_size
        )
        if self.args.get('freeze_input_proj'):
            for param in self.llama_proj.parameters():
                param.requires_grad = False

        # CS / SD modules
        hidden_size = self.llama_model.config.hidden_size
        self.cs = ContentSynchronizer(
            d_in=hidden_size,
            d_latent=int(args.get('d_latent_cs', 512)),
            d_out=int(args.get('d_out', 768)),
            num_layers=int(args.get('num_layers', 4)),
            nhead=int(args.get('nhead', 8)),
            dim_ff=int(args.get('dim_ff', 2048)),
        )
        self.sd = StyleDisentangler(
            d_in=hidden_size,
            d_latent=int(args.get('d_latent_sd', 256)),
            d_out=int(args.get('d_out', 768)),
            num_layers=int(args.get('num_layers', 4)),
            nhead=int(args.get('nhead', 8)),
            dim_ff=int(args.get('dim_ff', 2048)),
        )

        # loss weights
        self.alpha = float(args.get('alpha', 0.3))
        self.beta = float(args.get('beta', 0.3))
        self.kld_w = float(args.get('kld_weight', 1.0e-4))

        # Gold encoders (frozen) – initialized only when checkpoint paths are supplied
        styletts2_ckpt = args.get('styletts2_ckpt_dir')
        if styletts2_ckpt:
            self.styletts2 = StyleTTS2Encoders(styletts2_ckpt).to(self.device)
            for p in self.styletts2.parameters():
                p.requires_grad = False
            self.styletts2.eval()
        else:
            self.styletts2 = None

        dreamtalk_ckpt = args.get('dreamtalk_ckpt_dir')
        if dreamtalk_ckpt:
            self.dreamtalk = KeyFaceEncoders(dreamtalk_ckpt).to(self.device)
            for p in self.dreamtalk.parameters():
                p.requires_grad = False
            self.dreamtalk.eval()
        else:
            self.dreamtalk = None


    def _init_imagebind(self):
        imagebind_ckpt_path = os.path.join(self.args['pretrained_ckpt_path'], 'imagebind_ckpt',
                                           self.args['imagebind_version'])
        print(f'Initializing visual encoder from {imagebind_ckpt_path} ...')
        self.visual_encoder, self.visual_hidden_size = \
            imagebind_model.imagebind_huge(pretrained=True, store_path=imagebind_ckpt_path)
        # free vision encoder
        for name, param in self.visual_encoder.named_parameters():
            param.requires_grad = False
        self.visual_encoder.eval()
        print('Visual encoder initialized.')

    def _init_language_model(self):
        self.vicuna_ckpt_path = os.path.join(self.args['pretrained_ckpt_path'], 'vicuna_ckpt',
                                             self.args['vicuna_version'])
        print(f'Initializing language decoder from {self.vicuna_ckpt_path} ...')

        self.llama_model = LlamaForCausalLM.from_pretrained(self.vicuna_ckpt_path)
        
        #add the lora module
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.args['lora_r'],
            lora_alpha=self.args['lora_alpha'],
            lora_dropout=self.args['lora_dropout'],
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']
        )

        self.llama_model = get_peft_model(self.llama_model, peft_config)
        self.llama_model.print_trainable_parameters()

        if self.args.get('freeze_lm'):
            print("Freezing the LLaMa ...")
            for param in self.llama_model.parameters():
                param.requires_grad = False
            self.llama_model.eval()
        else:
            print('Language decoder initialized.')

        # use the new trained tokenizer
        tokenizer_path = self.vicuna_ckpt_path
        print(f'Initializing tokenizer from {tokenizer_path} ...')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path, use_fast=False)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        self.llama_tokenizer.padding_side = "right"

    # def _add_video_token(self):
    #     self.llama_tokenizer.add_tokens('<Vid>')  

    #     # Add [VID] tokens to the vocabulary.
    #     self.args['gen_video_token_idx'] = []
    #     for i in range(self.args['num_gen_vid_tokens']):
    #         print(f'Adding [VID{i}] token to vocabulary.')
    #         print(f'Before adding new token, tokenizer("[VID{i}]") =',
    #               self.llama_tokenizer(f'[VID{i}]', add_special_tokens=False))
    #         num_added_tokens = self.llama_tokenizer.add_tokens(f'[VID{i}]')
    #         print(f'After adding {num_added_tokens} new tokens, tokenizer("[VID{i}]") =',
    #               self.llama_tokenizer(f'[VID{i}]', add_special_tokens=False))
    #         gen_token_idx = self.llama_tokenizer(f'[VID{i}]', add_special_tokens=False).input_ids
    #         assert len(gen_token_idx) == 1, gen_token_idx
    #         self.args['gen_video_token_idx'].append(gen_token_idx[0])

    # def _add_audio_token(self):
    #     self.llama_tokenizer.add_tokens('<Aud>')  # add special audio token to tokenizer

    #     # Add [AUD] tokens to the vocabulary.
    #     self.args['gen_audio_token_idx'] = []
    #     for i in range(self.args['num_gen_aud_tokens']):
    #         print(f'Adding [AUD{i}] token to vocabulary.')
    #         print(f'Before adding new token, tokenizer("[AUD{i}]") =',
    #               self.llama_tokenizer(f'[AUD{i}]', add_special_tokens=False))
    #         num_added_tokens = self.llama_tokenizer.add_tokens(f'[AUD{i}]')
    #         print(f'After adding {num_added_tokens} new tokens, tokenizer("[AUD{i}]") =',
    #               self.llama_tokenizer(f'[AUD{i}]', add_special_tokens=False))
    #         gen_token_idx = self.llama_tokenizer(f'[AUD{i}]', add_special_tokens=False).input_ids
    #         assert len(gen_token_idx) == 1, gen_token_idx
    #         self.args['gen_audio_token_idx'].append(gen_token_idx[0])

    def encode_video(self, inputs):
        input_video_embs_list = []
        video_llama_atts_list = []

        dia_ids = inputs['dia_ids']
        max_utt_ids = []
        for dia in inputs['conversations']:
            max_utt_ids.append(len(dia['dialogue_history']))
            
        for i in range(len(dia_ids)):
            video_paths = []
            for utt_id in range(max_utt_ids[i]):
                dia_str = str(dia_ids[i]).zfill(5)
                pattern = os.path.join(self.args['video_path'], f'dia{dia_str}utt{utt_id + 1}_[0-9]*.mp4')
                video_pathes = glob.glob(pattern)
                if not video_pathes:
                    raise FileNotFoundError(f"No video file found for pattern: {pattern}")
                video_paths.append(video_pathes[0])
            inputs = {ModalityType.VISION: data.load_and_transform_video_data(video_paths, self.device)}
                # convert into visual dtype
            inputs = {key: inputs[key].to(self.llama_model.dtype) for key in inputs}
            with torch.no_grad():
                embeddings = self.visual_encoder(inputs)
                video_embeds = embeddings[ModalityType.VISION]  
                input_video_embs = self.llama_proj(video_embeds)
                video_llama_atts = torch.ones(input_video_embs[0].size()[:-1], dtype=torch.long).to(self.device) 
            input_video_embs_list.append(input_video_embs)
            video_llama_atts_list.append(video_llama_atts)
        return input_video_embs_list, video_llama_atts


    def encode_audio(self, inputs):
        input_audio_embs_list = []
        audio_llama_atts_list = []

        dia_ids = inputs['dia_ids']
        max_utt_ids = []
        for dia in inputs['conversations']:
            max_utt_ids.append(len(dia['dialogue_history']))
            
        for i in range(len(dia_ids)):
            audio_paths = []
            for utt_id in range(max_utt_ids[i]):
                dia_str = str(dia_ids[i]).zfill(5)
                pattern = os.path.join(self.args['audio_path'], f'dia{dia_str}utt{utt_id + 1}_[0-9]*.wav')
                audio_pathes = glob.glob(pattern)
                if not audio_pathes:
                    raise FileNotFoundError(f"No audio file found for pattern: {pattern}")
                audio_paths.append(audio_pathes[0])
            inputs = {ModalityType.AUDIO: data.load_and_transform_audio_data(audio_paths, self.device)}
            # convert into visual dtype
            inputs = {key: inputs[key].to(self.llama_model.dtype) for key in inputs}
            with torch.no_grad():
                embeddings = self.visual_encoder(inputs)
                audio_embeds = embeddings[ModalityType.AUDIO]  
                input_audio_embs = self.llama_proj(audio_embeds)  #  [1,4096],[3，4096]...
                audio_llama_atts = torch.ones(input_audio_embs.size()[:-1], dtype=torch.long).to(self.device)  
            input_audio_embs_list.append(input_audio_embs)
            audio_llama_atts_list.append(audio_llama_atts)
        return input_audio_embs_list, audio_llama_atts 

    def prompt_wrap(self, inputs_audio_embs, inputs_video_embs,  input_ids, target_ids, attention_mask):
    
        batch_size = input_ids.shape[0]
        audio_bos_id = self.llama_tokenizer('<Aud>', add_special_tokens=False).input_ids
        video_bos_id = self.llama_tokenizer('<Vid>', add_special_tokens=False).input_ids

        bos = torch.ones([batch_size, 1], dtype=input_ids.dtype,
                         device=input_ids.device) * self.llama_tokenizer.bos_token_id  

        p_after_embeds = self.llama_model.model.model.embed_tokens(input_ids).expand(batch_size, -1, -1)  
        bos_embeds = self.llama_model.model.model.embed_tokens(bos)  
        if inputs_audio_embs is not None and inputs_video_embs is not None:            
            audio_pos_list = [] 
            video_pos_list = [] 
            for b in range(input_ids.size(0)): 
                audio_pos = []
                video_pos = []
                for i, id in enumerate(input_ids[b]):
                    if id == audio_bos_id[0]:
                        audio_pos.append(i)
                    if id == video_bos_id[0]:
                        video_pos.append(i)
                assert len(audio_pos) == inputs_audio_embs[b].size(0)
                audio_pos_list.append(audio_pos)
                video_pos_list.append(video_pos)

        for b in range(input_ids.size(0)):
            audio_pos, video_pos = audio_pos_list[b], video_pos_list[b]
            for p in range(len(audio_pos)):
                p_after_embeds[b][audio_pos[p], :] = inputs_audio_embs[b][p]
                p_after_embeds[b][video_pos[p], :] = inputs_video_embs[b][p]
                
        inputs_embeds = torch.cat((bos_embeds, p_after_embeds), dim=1)  
        att = torch.ones([input_ids.size(0), 1],dtype=input_ids.dtype,
                         device=input_ids.device)
        attention_mask = torch.cat((att, attention_mask), dim=1)

        empty_targets = (torch.ones([batch_size, 1],  dtype=torch.long).to(self.device).fill_(-100))  
        targets = torch.cat([empty_targets, target_ids], dim=1).to(self.device) 
        return inputs_embeds, targets, attention_mask

    def _extract_signals(self, hidden, input_ids, audio_token_id, video_token_id):
        """
        hidden: [B, S, D] last hidden states from LLaMA
        input_ids: [B, S] token ids used to find <Aud>/<Vid> positions
        Returns r_t, r_s, r_v as variable-length sequences per batch (padded to max len in-batch).
        """
        B, S, D = hidden.shape
        device = hidden.device

        # masks
        aud_mask = (input_ids == audio_token_id)
        vid_mask = (input_ids == video_token_id)
        # consider "text tokens" as those that are NOT <Aud>/<Vid> and NOT padding (-100)
        txt_mask = (~aud_mask) & (~vid_mask) & (input_ids != -100)

        # simple gather: keep positions where mask True; pad to max length per mask type
        def gather(mask):
            max_len = mask.sum(dim=1).max().item()
            if max_len == 0:
                # fallback: at least BOS position (index 0)
                return hidden[:, :1, :]
            outs = []
            for b in range(B):
                idx = mask[b].nonzero(as_tuple=False).squeeze(-1)
                if idx.numel() == 0:
                    outs.append(hidden[b:b+1, :1, :])  # [1,1,D]
                else:
                    outs.append(hidden[b:b+1, idx, :])  # [1,Lb,D]
            # pad to [B, max_len, D]
            import torch.nn.functional as F
            outs = [F.pad(x, (0,0,0,int(max_len - x.shape[1]))) for x in outs]
            return torch.cat(outs, dim=0)

        r_t = gather(txt_mask)  # [B, Lt, D]
        r_s = gather(aud_mask)  # [B, La, D]
        r_v = gather(vid_mask)  # [B, Lv, D]
        return r_t, r_s, r_v

    def _load_golds(self, inputs, last_only=True):
        """
        Build gold targets for alignment from your file layout.
        - For content-text gold (C_s_gold): use gold response text strings.
        - For content-audio gold (C_v_gold): use response (or last-utterance) audio waveforms.
        - For style-speech gold (S_s_gold): same waveforms.
        - For style-video gold (S_v_gold): use response (or last-utterance) video clip tensors.
        """
        # ---- C_s_gold: gold response text ----
        try:
            targets = [dia['response_text'] for dia in inputs['conversations']]
        except Exception:
            # fallback if dataset stores strings directly
            targets = inputs['conversations']

        # ---- audio/video paths for RESPONSE (or last utterance) ----
        dia_ids = inputs['dia_ids']
        max_utt_ids = [len(dia['dialogue_history']) for dia in inputs['conversations']]
        # choose the last utterance as proxy for response media if explicit response media isn’t stored
        utt_index = -1 if last_only else 0

        aud_paths_batch = []
        vid_paths_batch = []
        for i, did in enumerate(dia_ids):
            ucount = max_utt_ids[i]
            u = ucount if utt_index == -1 else 1  # last or first
            aud_paths_batch.append(os.path.join(self.args['audio_path'], f"dia{did}utt{u}.wav"))
            vid_paths_batch.append(os.path.join(self.args['video_path'], f"dia{did}utt{u}.mp4"))

        # ---- read audio to tensors (concat or pad) ----
        wavs = []
        for p in aud_paths_batch:
            try:
                wav, sr = sf.read(p)
                wav = torch.from_numpy(wav).float().to(self.device)
            except Exception:
                wav = torch.zeros(16000, device=self.device)  # 1s silence fallback to keep graph valid
            wavs.append(wav)
        # pad to the longest
        max_w = max(w.shape[0] for w in wavs)
        wavs = [F.pad(w, (0, max_w - w.shape[0])) for w in wavs]
        wav_batch = torch.stack(wavs, dim=0)  # [B, T]

        # ---- read short video tensors: [B, T, C, H, W] ----
        vids = []
        for p in vid_paths_batch:
            frames = []
            cap = cv2.VideoCapture(p)
            ok = True; t = 0
            while ok and t < 16:  # 16 frames is plenty for a style encoder
                ok, fr = cap.read()
                if ok:
                    fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                    fr = torch.from_numpy(fr).permute(2,0,1).float()/255.0  # [C,H,W]
                    frames.append(fr)
                    t += 1
            cap.release()
            if not frames:
                frames = [torch.zeros(3, 128, 128)]
            vid = torch.stack(frames, dim=0)  # [T,C,H,W]
            vids.append(vid)
        # pad T
        max_t = max(v.shape[0] for v in vids)
        vids = [F.pad(v, (0,0,0,0,0,0,0, max_t - v.shape[0])) for v in vids]
        vid_batch = torch.stack(vids, dim=0).to(self.device)  # [B, T, C, H, W]

        # ---- run encoders (no grad) ----
        B = len(targets)
        d_out = int(self.args.get('d_out', 768))
        with torch.no_grad():
            if self.styletts2 is not None:
                C_s_gold = self.styletts2.text_content(targets).to(self.device)
                S_s_gold = self.styletts2.style_from_audio(wav_batch)
            else:
                C_s_gold = torch.zeros(B, d_out, device=self.device)
                S_s_gold = torch.zeros(B, d_out, device=self.device)

            if self.dreamtalk is not None:
                C_v_gold = self.dreamtalk.content_from_audio(wav_batch)
                S_v_gold = self.dreamtalk.style_from_video(vid_batch)
            else:
                C_v_gold = torch.zeros(B, d_out, device=self.device)
                S_v_gold = torch.zeros(B, d_out, device=self.device)
        return C_s_gold, C_v_gold, S_s_gold, S_v_gold, targets

    def _empathetic_diallogue_training(self, target_ids, outputs):
        """
        In the stage 1: training the text-based empathetic response generation ability via EmpatheticDialogue dataset
        """

        loss = outputs.loss
        # calculate the token accuracy
        chosen_tokens = torch.max(outputs.logits, dim=-1)[1][:,1:-1] # [B, S-1]
        labels = target_ids[:,2:]
        gen_acc = (chosen_tokens.reshape(-1) == labels.reshape(-1)).to(torch.long)  # [B*S]
        valid_mask = (labels != -100).reshape(-1)
        valid_tokens = gen_acc & valid_mask  # [B*S]
        gen_acc = valid_tokens.sum().item() / (valid_mask.sum().item() + 1.0)
        return loss, gen_acc


    def forward(self, inputs): # 안 씀, forward_llm 씀
        gen_acc = 0

        input_ids, target_ids, attention_mask = process_batch_text_stream(self.llama_tokenizer,
                                                inputs['conversations'],
                                                self.max_length
                                                )
        input_ids = input_ids.to(self.device)  
        target_ids = target_ids.to(self.device) 
        attention_mask = attention_mask.to(self.device)
        
        inputs_audio_embs, audio_llama_atts = self.encode_audio(inputs)
        inputs_video_embs, video_llama_atts = self.encode_video(inputs)
        inputs_embeds, target_ids, attention_mask = self.prompt_wrap(
            inputs_audio_embs, 
            inputs_video_embs, 
            input_ids, 
            target_ids, 
            attention_mask
            )

        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
            labels=target_ids
        )
        llama_loss, gen_acc = self._empathetic_diallogue_training(target_ids, outputs)

        # ===== CS/SD integration starts here =====
        # Hidden states (exclude the very last token if you want strictly "inputs" only)
        hidden = outputs.hidden_states[-1]  # [B, S, D]
        audio_token_id = self.llama_tokenizer('<Aud>', add_special_tokens=False).input_ids[0]
        video_token_id = self.llama_tokenizer('<Vid>', add_special_tokens=False).input_ids[0]

        # r_t (text), r_s (speech markers), r_v (video markers)
        r_t, r_s, r_v = self._extract_signals(hidden, input_ids, audio_token_id, video_token_id)

        # CS: content heads
        C_s, C_v, kld_cs = self.cs(r_t)  # -> [B, 768], [B, 768], scalar KL

        # SD: style heads + classification logits
        S_s, S_v, logits, kld_sd = self.sd(r_s, r_v)

        # Golds from encoders + labels from batch
        C_s_gold, C_v_gold, S_s_gold, S_v_gold, targets = self._load_golds(inputs)
        labels = {
            'emotion': inputs['response_emotion'].to(self.device),
            'age': inputs['response_profile']['age'].to(self.device),
            'gender': inputs['response_profile']['gender'].to(self.device),
            'tone': (inputs['response_profile'].get('timbre', None) or inputs['response_profile']['tone']).to(
                self.device)
        }

        L_ccl = loss_ccl(C_s, C_v, C_s_gold, C_v_gold)
        L_sal = loss_sal(S_s, S_v, S_s_gold, S_v_gold)
        L_cls = loss_cls(logits, labels)
        loss_total = llama_loss + self.alpha * L_ccl + self.beta * (L_sal + L_cls) + self.kld_w * (kld_cs + kld_sd)
        # ===== CS/SD integration ends here =====

        return {
            'gen_acc': gen_acc,
            'loss': loss_total,
            'loss_llm': llama_loss.detach(),
            'loss_ccl': L_ccl.detach(),
            'loss_sal': L_sal.detach(),
            'loss_cls': L_cls.detach()
        }

    def forward_llm(self, inputs):
        input_ids, target_ids, attention_mask = process_batch_text_stream(self.llama_tokenizer,
                                                                          inputs['conversations'],
                                                                          self.max_length
                                                                          )
        input_ids = input_ids.to(self.device)
        target_ids = target_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        inputs_audio_embs, audio_llama_atts = self.encode_audio(inputs)
        inputs_video_embs, video_llama_atts = self.encode_video(inputs)
        inputs_embeds, target_ids, attention_mask = self.prompt_wrap(
            inputs_audio_embs,
            inputs_video_embs,
            input_ids,
            target_ids,
            attention_mask
        )

        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
            labels=target_ids
        )

        return outputs, inputs_embeds, input_ids, target_ids, attention_mask

    def return_generate(self, inputs):
        input_ids, target_ids, attention_mask = process_batch_text_stream(self.llama_tokenizer,
                                                                          inputs['conversations'],
                                                                          self.max_length
                                                                          )

        input_ids = input_ids.to(self.device)
        target_ids = target_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        inputs_audio_embs, audio_llama_atts = self.encode_audio(inputs)
        inputs_video_embs, video_llama_atts = self.encode_video(inputs)
        inputs_embeds, target_ids, attention_mask = self.prompt_wrap(
            inputs_audio_embs,
            inputs_video_embs,
            input_ids,
            target_ids,
            attention_mask
        )

        return inputs_embeds, attention_mask

    @torch.no_grad()
    def generate_response(
            self,
            inputs,
            max_new_tokens=128,
            temperature=0.8,
            top_p=0.95,
            do_sample=True
    ):
        """
        멀티모달 입력 또는 텍스트-only 입력을 받아 실제 생성결과(문장)를 리턴.
        """
        self.eval()
        # ----- (1) 텍스트 tokenize (풀 문장 입력)
        input_ids, _, attention_mask = process_batch_text_stream(
            self.llama_tokenizer,
            inputs['conversations'],
            self.max_length
        )
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        # ----- (2) 멀티모달 임베딩 추출 (오디오/비디오 있을 때만)
        if 'use_multimodal' in inputs and inputs['use_multimodal']:
            inputs_audio_embs, audio_llama_atts = self.encode_audio(inputs)
            inputs_video_embs, video_llama_atts = self.encode_video(inputs)
        else:
            inputs_audio_embs = None
            inputs_video_embs = None

        # ----- (3) 입력 임베딩 조립
        if (inputs_audio_embs is not None and inputs_video_embs is not None
                and len(inputs_audio_embs) > 0 and len(inputs_video_embs) > 0):
            # Multimodal: build combined input embeddings and call generate with inputs_embeds.
            # HuggingFace GenerationMixin supports inputs_embeds as a direct replacement for
            # input_ids so audio/video tokens are naturally included in the context.
            inputs_embeds, _, attention_mask = self.prompt_wrap(
                inputs_audio_embs,
                inputs_video_embs,
                input_ids,
                torch.zeros_like(input_ids),  # targets not needed for inference
                attention_mask
            )
            generated_ids = self.llama_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample
            )
            output_text = self.llama_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            return output_text
        else:
            # Text-only inference
            generated_ids = self.llama_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample
            )
            output_text = self.llama_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            return output_text

    

    
