import argparse
import torch
from transformers import LlamaTokenizer, LlamaConfig
from model import *
from dataset import load_dataset
from config import load_config
from header import *
import os

def parser_args():
    parser = argparse.ArgumentParser(description='inference script')
    parser.add_argument('--model', type=str, default='merg')
    parser.add_argument('--weight_path', type=str, default='ckpt/pretrained_ckpt/vicuna_ckpt/7b_v0')
    parser.add_argument('--audio_path', type=str, default='/mnt/dataset/AvaMERG_jhchoi/AvaMERG/audio_v5_0')
    parser.add_argument('--video_path', type=str, default="/mnt/dataset/AvaMERG_jhchoi/AvaMERG/video_v5_0")
    parser.add_argument('--max_length', type=int, default=1024)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--ds_config_path', type=str, default='merg_code/dsconfig/dsconfig.json')  # train 코드 참고
    parser.add_argument('--log_path', type=str, default='./logs')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--save_path', type=str, default='ckpt/merg_ckpt2/')
    parser.add_argument('--epochs', type=int, default=2)  # 체크포인트 폴더명
    parser.add_argument('--total_steps', type=int, default=1)
    parser.add_argument('--warmup_rate', type=float, default=0.1)
    parser.add_argument('--logging_step', type=int, default=100)
    return parser.parse_args()

def main():
    args = parser_args()
    args = vars(args)

    device = torch.device(args['device'])

    # load config if 추가 로드 필요 시
    args = load_config(args)

    # Load tokenizer and config from saved checkpoint folder
    tokenizer = LlamaTokenizer.from_pretrained(args['weight_path'])
    config = LlamaConfig.from_pretrained(args['weight_path'])

    # Load model architecture via load_model from project (train 코드 참고)
    agent = load_model(args)

    # Load test dataset following train 코드 방식
    test_data, test_iter, sampler = load_dataset(args)
    # test_iter = [1]

    max_samples = 1000
    total_loss, total_acc, total_count = 0, 0, 0
    current_step = 0
    pbar = tqdm(total=max_samples)

    '''ver1: loss'''
    # for batch in test_iter:
    #     batch_size = 1
    #     if total_count + batch_size > max_samples:
    #         batch_size = max_samples - total_count  # 남은 소수 샘플만 사용
    #     loss, acc = agent.test_model(batch, current_step=current_step, pbar=pbar)
    #     total_loss += loss * batch_size
    #     total_acc += acc * batch_size
    #     total_count += batch_size
    #     current_step += 1
    #     if total_count >= max_samples:
    #         break
    #
    # if total_count > 0:
    #     mean_loss = total_loss / total_count
    #     mean_acc = total_acc / total_count
    #     print(f"\n[Test] (최대 {max_samples}개 샘플 기준) 평균 Loss: {mean_loss:.4f} | 평균 Accuracy: {mean_acc:.2f}")
    # else:
    #     print("테스트 데이터가 비었습니다.")

    '''ver2: prompt'''
    for batch in test_iter:
        inputs_embeds, attention_mask = agent.inference_model(batch)


if __name__ == "__main__":
    main()
