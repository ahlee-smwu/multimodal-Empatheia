from header import *
from dataset import load_dataset
from model import *
from config import load_config

from accelerate import Accelerator, FullyShardedDataParallelPlugin
import torch
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader

def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--model', type=str, default='merg')
    parser.add_argument('--mode', type=str, default='train', help='train or test')
    parser.add_argument('--audio_path', type=str, default="/mnt/dataset/AvaMERG_jhchoi/AvaMERG/audio_v5_0")
    parser.add_argument('--video_path', type=str, default="/mnt/dataset/AvaMERG_jhchoi/AvaMERG/video_v5_0")
    parser.add_argument('--save_path', type=str, default='ckpt/merg_ckpt/')
    parser.add_argument('--log_path', type=str, default='ckpt/merg_ckpt/')
    parser.add_argument('--assets_path', type=str, default='./assets/')
    parser.add_argument('--max_length', type=int, default=1024)
    parser.add_argument('--fp16', action='store_true', help='use mixed precision')
    parser.add_argument('--lr', type=float, default=1e-4, help='optimizer lr')
    parser.add_argument('--grad_accum_steps', type=int, default=8, help='gradient accumulation steps')
    parser.add_argument('--batch_size', type=int, default=1, help='per GPU batch size')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()

def set_random_seed(seed):
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.random.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

def build_directory(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def main(**args):
    args = load_config(args)
    args.setdefault('world_size', int(os.getenv('WORLD_SIZE', '1')))
    print("Loaded args/config:", args)

    # -------------------------
    # Gradient Checkpointing, FSDP
    # -------------------------
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    # Accelerator + FSDP Plugin
    policy = transformer_auto_wrap_policy
    accelerator = Accelerator(
        mixed_precision='fp16' if args.get('fp16', False) else 'no',
        fsdp_plugin=FullyShardedDataParallelPlugin(auto_wrap_policy=policy)
    )
    print("Accelerator initialized:", accelerator.state)

    set_random_seed(args.get('seed', 42))
    build_directory(args['save_path'])
    build_directory(args['log_path'])

    # logging
    if args['log_path']:
        logging.basicConfig(
            format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
            level=logging.DEBUG,
            filename=f'{args["log_path"]}/train_{time.asctime()}.log',
            filemode='w'
        )

    # -------------------------
    # Dataset
    # -------------------------
    train_data, train_iter, sampler = load_dataset(args)
    train_num = len(train_data)
    print(f'Num of training data: {train_num}')

    # -------------------------
    # Model / Agent
    # -------------------------
    agent = load_model(args)

    # enable gradient checkpointing if supported
    if hasattr(agent.model, "gradient_checkpointing_enable"):
        agent.model.gradient_checkpointing_enable()

    # optimizer
    if hasattr(agent, 'optimizer') and agent.optimizer is not None:
        optimizer = agent.optimizer
    else:
        optimizer = torch.optim.AdamW(agent.model.parameters(), lr=args.get('lr', 1e-4))

    # wrap with accelerator (FSDP + mixed precision)
    agent.model, optimizer, train_iter = accelerator.prepare(agent.model, optimizer, train_iter)
    agent.optimizer = optimizer
    agent.accelerator = accelerator
    agent.model_device = accelerator.device

    # -------------------------
    # Training loop
    # -------------------------
    effective_batch = args.get('batch_size', 1) * args.get('grad_accum_steps', 1)
    length = args['epochs'] * train_num // effective_batch
    total_steps = args['epochs'] * train_num // effective_batch
    args['total_steps'] = total_steps

    pbar = tqdm(total=length) if accelerator.is_main_process else None
    current_step = 0

    for epoch_i in range(args['epochs']):
        agent.model.train()
        for batch in train_iter:
            # gradient accumulation
            with accelerator.accumulate(agent.model):
                agent.train_model(batch, current_step=current_step, pbar=pbar)
            current_step += 1

        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            try:
                save_obj = accelerator.unwrap_model(agent.model)
            except Exception:
                save_obj = agent.model
            agent.save_model(args['save_path'], epoch_i+1, current_step)

    if pbar is not None:
        pbar.close()

if __name__ == "__main__":
    args = parser_args()
    args = vars(args)
    main(**args)
