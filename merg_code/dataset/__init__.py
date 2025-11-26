from header import *
from .samplers import DistributedBatchSampler
import torch
import importlib
from dataset.all_dataset import multimodal_empathetic_dialogue


def load_dataset(args):
    _dataset = multimodal_empathetic_dialogue(args)

    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    if args['mode'] == 'train':
        # batch_size = args['world_size'] * args['dschf'].config['train_micro_batch_size_per_gpu']
        batch_size = 1 #!!!!!!!!!!!!!!!!!! just for test, delete it for train
    elif args['mode'] == 'test':
        batch_size = 1
    else:
        raise ValueError(" Mode Error! The mode should be train or test! ")
    sampler = torch.utils.data.RandomSampler(_dataset)
    batch_sampler = DistributedBatchSampler(
                                                sampler=sampler,
                                                batch_size=batch_size,
                                                drop_last=True,
                                                rank=rank,
                                                world_size=world_size,
                                            )
    '''original iter'''
    iter_ = DataLoader(
        _dataset,
        batch_sampler=batch_sampler,
        num_workers=1,
        collate_fn=_dataset.collate_fn,
        pin_memory=True
    )
    # iter_ = DataLoader(
    #     _dataset,
    #     batch_size=args.get('batch_size', 1),
    #     shuffle=True if args['mode'] == 'train' else False,
    #     collate_fn=_dataset.collate_fn,
    #     num_workers=1,
    #     pin_memory=True
    # )
    return _dataset, iter_, None
