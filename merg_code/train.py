from header import *
from dataset import load_dataset
from model import *
from config import load_config

os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'

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

    return parser.parse_args()


def initialize_distributed(args):
    args['master_ip'] = os.getenv('MASTER_ADDR', 'localhost')
    args['master_port'] = os.getenv('MASTER_PORT', '6000')
    args['world_size'] = int(os.getenv('WORLD_SIZE', '1'))
    args['local_rank'] = int(os.getenv('RANK', '0')) % torch.cuda.device_count()
    device = args['local_rank'] % torch.cuda.device_count()
    torch.cuda.set_device(device)
    deepspeed.init_distributed(dist_backend='nccl')


def set_random_seed(seed):
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def build_directory(path):
    if os.path.exists(path):
        pass
    else: 
        os.makedirs(path, exist_ok=True)


def main(**args):
    args = load_config(args)
    print(args)
    initialize_distributed(args)
    set_random_seed(args['seed'])
    args['ds_config_path'] = f'merg_code/dsconfig/dsconfig.json'
    dschf = HfDeepSpeedConfig(args['ds_config_path'])
    args['dschf'] = dschf

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

    train_num = train_data.__len__()
    print(f'################################# Num of training data #######################################: {train_num}')
    length = args['epochs'] * train_num // args['world_size'] // dschf.config[
        'train_micro_batch_size_per_gpu']
    total_steps = args['epochs'] * train_num // dschf.config['train_batch_size']
    args['total_steps'] = total_steps
    agent = load_model(args)
    torch.distributed.barrier()

    pbar = tqdm(total=length)  
    current_step = 0


    for epoch_i in tqdm(range(args['epochs'])):
        for batch in train_iter:
            agent.train_model(
                batch, 
                current_step=current_step,
                pbar=pbar
            )
            current_step += 1
        torch.distributed.barrier()
        agent.save_model(args['save_path'], epoch_i+1, current_step)
    


if __name__ == "__main__":
    args = parser_args()
    args = vars(args)
    main(**args)

