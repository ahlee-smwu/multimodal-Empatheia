import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from dataset.all_dataset import multimodal_empathetic_dialogue


def load_dataset(args):
    dataset = multimodal_empathetic_dialogue(args)

    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()

    if args['mode'] == 'train':
        batch_size = args['dschf'].config['train_micro_batch_size_per_gpu']
        shuffle = True
    elif args['mode'] == 'test':
        batch_size = 1
        shuffle = False
    else:
        raise ValueError("Mode Error! The mode should be train or test!")

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        drop_last=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=min(8, max(1, torch.get_num_threads() // world_size)),
        collate_fn=dataset.collate_fn,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    return dataset, dataloader, sampler
