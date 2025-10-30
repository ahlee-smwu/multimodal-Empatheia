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
import glob
# from speech_generator.generate_audio import StyleTTS2
# from talking_face_generator.generate_video import generate_video

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
                    raise FileNotFoundError(f"No audio file found for pattern: {pattern}")
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


    def forward(self, inputs):
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
        return {
            'gen_acc': gen_acc,
            'loss':llama_loss
        }
    

    
