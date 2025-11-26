import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from tqdm import tqdm
from header import *
from model.common.utils import *
from transformers import LlamaForCausalLM, LlamaConfig, LlamaTokenizer
from config import load_config

ckpt_dir = "ckpt/merg_ckpt/6" #"ckpt/pretrained_ckpt/vicuna_ckpt/7b_v0"

'''convert to model.bin'''
# print("STEP 1: Load config & tokenizer")
# config = LlamaConfig.from_pretrained(ckpt_dir)
# tokenizer = LlamaTokenizer.from_pretrained(ckpt_dir)
#
# print("STEP 2: Build base model object")
# model = LlamaForCausalLM(config)
#
# print("STEP 3: Load state_dict from pytorch_model.pt")
# state_dict = torch.load(f"{ckpt_dir}/pytorch_model.pt", map_location="cpu")
# model.load_state_dict(state_dict, strict=False)
#
# print("STEP 4: Save as Huggingface format (pytorch_model.bin)")
# model.save_pretrained(ckpt_dir)
# tokenizer.save_pretrained(ckpt_dir)
#
# print("변환 완료! 이제 from_pretrained로 모델을 정상적으로 불러올 수 있습니다.")


print("STEP 1: 모델 & 토크나이저 불러오기")
tokenizer = LlamaTokenizer.from_pretrained(ckpt_dir)
model = LlamaForCausalLM.from_pretrained(ckpt_dir).to("cuda" if torch.cuda.is_available() else "cpu").eval()

print("Special tokens loaded:", tokenizer.special_tokens_map)
if hasattr(tokenizer, 'added_tokens_encoder'):
    print("Added tokens:", list(tokenizer.added_tokens_encoder.keys()))

# 1. 프롬프트(실제 사용자 컨텍스트) dict 작성
conversations = {
    "dialogue_history": [
        {"utterance": "Hi, how are you doing today?"},
        {"utterance": "I'm not feeling well... Everything seems tough lately."},
        {"utterance": "I'm really sorry to hear that. Do you want to talk about it?"}
    ],
    "coe": {
        "event_scenario": "Daily life stress",
        "speaker_emotion": "Sadness",
        "emotion_cause": "Overwhelming workload",
        "goal_to_response": "Emotional support and encouragement"
    },
    "response": "It's understandable to feel that way. You're not alone, and it's okay to take things one step at a time. If you need someone to listen, I'm here for you."
}


# 2. input_ids 생성 (타깃은 필요 X)
input_ids, _ = build_one_instance_text_stream(tokenizer, conversations)
input_ids = torch.tensor([input_ids], dtype=torch.long).to(model.device)

# 3. attention_mask 생성 (0은 패드, 1은 유효 토큰)
attention_mask = (input_ids != tokenizer.pad_token_id).long()


# 4. generate 호출 예시
model.eval()
with torch.no_grad():
    gen_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=128,
        temperature=0.8,
        top_p=0.95,
        do_sample=True
    )
gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=False)
print("추론 결과:", gen_text)

print("Inference done!")

'''tokenizer working <- done!'''
# # 1. 인코딩 과정
# my_prompt = "Use <Vid> and <Aud> token in sequence."
# token_ids = tokenizer(my_prompt, add_special_tokens=False)['input_ids']
# tokens = tokenizer.convert_ids_to_tokens(token_ids)
#
# print("== 토큰화 결과 ==")
# for tid, tok in zip(token_ids, tokens):
#     print(f"{tid}\t{tok}")
#
# # 2. 추가토큰 분리 여부 확인
# added_tokens = set(tokenizer.added_tokens_encoder.keys())
# used_additional = [tok for tok in tokens if tok in added_tokens]
# print("== 이 프롬프트에서 실제 사용된 추가토큰 ==")
# print(used_additional if used_additional else "없음")
#
# decoded_text = tokenizer.decode(token_ids, skip_special_tokens=False)
# print(decoded_text)

'''added_token working <- done!'''
# # Assume: tokenizer, model already loaded from the same ckpt_dir
# added_token_ids = list(tokenizer.added_tokens_encoder.values())
# embedding_size = model.get_input_embeddings().weight.shape[0]

# print(f"Tokenizer vocab size: {len(tokenizer)}")
# print(f"Model embedding matrix size: {embedding_size}")
# print("== 추가 토큰 id, 토큰명, 임베딩 매핑 유무 확인 ==")
# for token_str, token_id in tokenizer.added_tokens_encoder.items():
#     print(f"{token_str:<8}  id:{token_id:<6}  in range? {'YES' if token_id < embedding_size else 'NO'}")
# assert embedding_size == len(tokenizer), \
#     "임베딩 테이블과 토크나이저 vocab 길이가 일치하지 않습니다! (추가 토큰 임베딩이 미확장)"