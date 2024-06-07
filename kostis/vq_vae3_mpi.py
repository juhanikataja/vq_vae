#import sys
#sys.path.append('/scratch/project_462000599/kostis/libs/analysator')
from model_vq_vae import * 
from tools import *
import numpy as np
from tqdm import tqdm
import sys
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
from torch.multiprocessing import Process
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
#wrap ranks around GPU
local_rank = rank % torch.cuda.device_count()
torch.cuda.set_device(local_rank)
print(f"Task {rank} will torture device {rank % torch.cuda.device_count()} ")
assert torch.distributed.is_available(),"Not supported"
assert torch.distributed.is_nccl_available(),"Not supported"
assert size == torch.cuda.device_count(),"NCCL is in so we need this??" ;

comm.barrier()
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '9999'
os.environ['WORLD_SIZE'] = str(size)
os.environ['RANK'] = str(rank)
comm.barrier()

dist.init_process_group(backend='nccl', world_size=size, rank=rank)

filename=sys.argv[1]
device ='cuda'

# Initialize model
use_ema = True # Use exponential moving average
model_args = {
    "in_channels":1,
    "num_hiddens": 128,
    "num_downsampling_layers": 2,
    "num_residual_layers": 2,
    "num_residual_hiddens": 32,
    "embedding_dim": 64,
    "num_embeddings": 512,
    "use_ema": use_ema,
    "decay": 0.99,
    "epsilon": 1e-5,
}
model = VQVAE(**model_args).to(device)
model = DDP(model, device_ids=[local_rank])


batch_size = 5
workers = 0
cids = np.arange(1, 10)
VDF_Data = Vlasiator_DataSet(cids, filename, device)
train_sampler = DistributedSampler(VDF_Data, num_replicas=size, rank=rank)
train_loader = DataLoader(
    dataset=VDF_Data,
    batch_size=batch_size,
    sampler=train_sampler,
    num_workers=workers,
    pin_memory=False
)
# Multiplier for commitment loss. See Equation (3) in "Neural Discrete Representation Learning"
beta = 0.25

# Initialize optimizer
train_params = [params for params in model.parameters()]
lr = 3e-4
optimizer = optim.Adam(train_params, lr=lr)
criterion = nn.MSELoss()

# Train model
epochs = 3
eval_every = 1
best_train_loss = float("inf")
model.train()



for epoch in range(epochs):
    train_sampler.set_epoch(epoch)
    total_train_loss = 0
    total_recon_error = 0
    n_train = 0
    for batch_idx, train_tensors in enumerate(train_loader):
        optimizer.zero_grad()
        imgs = train_tensors[0].unsqueeze(0).to(local_rank)
        out = model(imgs)
        recon_error = criterion(out["x_recon"], imgs)
        total_recon_error += recon_error.item()
        loss = recon_error + beta * out["commitment_loss"]
        if not use_ema:
            loss += out["dictionary_loss"]

        total_train_loss += loss.item()
        loss.backward()
        optimizer.step()
        n_train += 1

        if ((batch_idx + 1) % eval_every) == 0 and rank == 0:
            print(f"epoch: {epoch}\nbatch_idx: {batch_idx + 1}", flush=True)
            avg_train_loss = total_train_loss / n_train
            if avg_train_loss < best_train_loss:
                best_train_loss = avg_train_loss

            print(f"best_train_loss: {best_train_loss}")
            print(f"recon_error: {total_recon_error / n_train}\n")
            total_train_loss = 0
            total_recon_error = 0
            n_train = 0
