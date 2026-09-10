from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel, CPUOffload
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

import torch


def setup_distributed_model(args, model):
    if args.distributed:
        if args.sharding:
            logging.info("Using FSDP for distributed training.")
            model = FullyShardedDataParallel(
                model,
                auto_wrap_policy=size_based_auto_wrap_policy,
                cpu_offload=CPUOffload(offload_params=True),
            )
        else:
            logging.info("Using DDP for distributed training.")
            model = DDP(
                model,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                find_unused_parameters=True,
            )
    return model
