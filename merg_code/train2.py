# train_accelerate.py
from header import *
from dataset import load_dataset
from model import *
from config import load_config

# remove deepspeed env forcing; rely on accelerate
# os.environ['RANK'] = '0'
# os.environ['WORLD_SIZE'] = '1'

from accelerate import Accelerator

def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--model', type=str, default='merg')
    parser.add_argument('--mode', type=str, default='train', help='train or test')
    parser.add_argument('--audio_path', type=str, default="/mnt/dataset/AvaMERG_jhchoi/AvaMERG/audio_v5_0")
    parser.add_argument('--video_path', type=str, default="/mnt/dataset/AvaMERG_jhchoi/AvaMERG/video_v5_0")
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--save_path', type=str, default='ckpt/merg_ckpt/')
    parser.add_argument('--log_path', type=str, default='ckpt/merg_ckpt/')
    parser.add_argument('--assets_path', type=str, default='./assets/')
    parser.add_argument('--max_length', type=int, default=1024)
    parser.add_argument('--fp16', action='store_true', help='use mixed precision')
    parser.add_argument('--lr', type=float, default=1e-4, help='optimizer lr (if not provided by agent)')
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
    if 'dschf' not in args or not hasattr(args['dschf'], 'config'):
        args['dschf'] = types.SimpleNamespace(config={'train_micro_batch_size_per_gpu': args.get('batch_size', 4)})

    accelerator = Accelerator(mixed_precision='fp16')
    set_random_seed(args.get('seed', 42))

    build_directory(args['save_path'])
    build_directory(args['log_path'])

    if args['log_path']:
        logging.basicConfig(
            format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
            level=logging.DEBUG,
            filename=f'{args["log_path"]}/train_{time.asctime()}.log',
            filemode='w'
        )

    train_data, train_iter, sampler = load_dataset(args)
    train_num = len(train_data)
    print(f'Num of training data: {train_num}')

    agent = load_model(args)

    # Create optimizer if agent doesn't provide one
    if hasattr(agent, 'optimizer') and agent.optimizer is not None:
        optimizer = agent.optimizer
    else:
        optimizer = torch.optim.AdamW(agent.model.parameters(), lr=args.get('lr', 1e-4))
    print("###################1")

    # Prepare model, optimizer, dataloader
    agent.model, optimizer, train_iter = accelerator.prepare(agent.model, optimizer, train_iter)
    print("###################2")
    agent.optimizer = optimizer
    agent.accelerator = accelerator
    agent.model_device = accelerator.device

    grad_accum_steps = args.get('gradient_accumulation_steps', 1)
    total_steps = args['epochs'] * train_num // max(1, args.get('train_batch_size', 1))
    args['total_steps'] = total_steps

    if accelerator.is_main_process:
        pbar = tqdm(total=total_steps)
    else:
        pbar = None

    current_step = 0
    print("###################3")

    for epoch_i in range(args['epochs']):
        agent.model.train()
        print("###################4")
        for batch in train_iter:
            print("###################5")
            with accelerator.accumulate(agent.model):
                # forward only
                loss, loss_dict = agent.forward_loss(batch)
                print("###################6")
                # backward
                accelerator.backward(loss)
                print("###################7")
                # step & zero_grad are handled by accumulate
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.is_main_process and pbar is not None:
                pbar.set_description(f"[Epoch {epoch_i + 1}] step {current_step}, loss={loss.item():.4f}")
                pbar.update(1)

            current_step += 1

        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            save_model_obj = accelerator.unwrap_model(agent.model)
            agent.save_model(args['save_path'], epoch_i + 1, current_step)

    if pbar is not None:
        pbar.close()

if __name__ == "__main__":
    args = parser_args()
    args = vars(args)
    main(**args)
