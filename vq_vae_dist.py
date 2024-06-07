#import sys

import sys
import os
from datetime import datetime
import psutil
#
import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
#from torch.utils.tensorboard import SummaryWriter

# Distributed
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
#from env_utils import print_slurm_env


import numpy as np
from tqdm import tqdm

sys.path.append('analysator')
import pytools as pt

# Import the model
from model_vq_vae import * 

# Import vdf tools
from tools import *

torch.cuda.empty_cache()

# init distributed

#torch.cuda.set_device(local_rank)

if __name__=="__main__":

  local_rank = int(os.environ['LOCAL_RANK'])
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])
  nodeid = int(os.environ["SLURM_NODEID"])
  print(f"rank/ranks: {rank}/{world_size}. Local rank: {local_rank}. Node-ID: {nodeid}")

  dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)

  cids=range(1,1000)
  filename=sys.argv[1]

  # input_array=extract_vdfs(filename,cids,25) # 25-> half the mesh dimension
  # input_array=input_array.squeeze();
  # print(input_array.shape)


  #device = 'cuda'
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu', local_rank)

  #device = 'cpu'
  print('Using device:', device)

  use_tb = True # Use Tensorboard (optional)

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

  print("Defining a model")
  model = DistributedDataParallel(VQVAE(**model_args).to(device),
                                  device_ids=[device])#, 
                                  # output_device=device,
                                  # )
  print("defined model")

  class VDFDataset():
      def __init__(self, cids,filename):
          self.cids=cids
          self.filename=filename

      def __len__(self):
          return len(self.cids)

      def __getitem__(self, idx):
          vdf=extract_vdf(self.filename, self.cids[idx],box=25)
          vdf_norm = (vdf - vdf.min())/(vdf.max() - vdf.min())
          return torch.tensor(vdf_norm).unsqueeze(0).to(device)
          

  # Initialize dataset
  batch_size = 2
  workers = 0

  # input_norm = (input_array - input_array.min())/(input_array.max() - input_array.min()) # MinMax normalization
  # input_tensor = torch.tensor(input_norm, dtype=torch.float32).unsqueeze(0)#.unsqueeze(0)#.to(device)  # Add batch and channel dimensions, move to device
  # print(input_tensor.shape)
  # train_dataset = input_tensor

  # VDF_Data=VDFDataset(cids,filename)
  VDF_Data=Vlasiator_DataSet(cids,filename, device)
  print("defined dataset")

  vdf_sampler=DistributedSampler(VDF_Data, num_replicas=world_size, rank=rank)
  print("defined sampler")

  train_loader = DataLoader(
      dataset=VDF_Data,
      sampler=vdf_sampler,
      batch_size=batch_size,
      shuffle=False,
      num_workers=workers,
      pin_memory=False,
  )
  print("defined dataloader")

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
  print("starting training..")
  # Training
  for epoch in tqdm(range(epochs)):
      train_sampler.set_epoch(epoch)
      total_train_loss = 0
      total_recon_error = 0
      n_train = 0
      for (batch_idx, train_tensors) in enumerate(train_loader):
          optimizer.zero_grad()
          imgs = train_tensors[0].unsqueeze(0).to(device)
          out = model(imgs)
          recon_error = criterion(out["x_recon"], imgs) #/ train_data_variance
          total_recon_error += recon_error.item()
          loss = recon_error + beta * out["commitment_loss"]
          if not use_ema:
              loss += out["dictionary_loss"]

          total_train_loss += loss.item()
          loss.backward()
          optimizer.step()
          n_train += 1

          if ((batch_idx + 1) % eval_every) == 0:
              print(f"epoch: {epoch}\nbatch_idx: {batch_idx + 1}", flush=True)
              total_train_loss /= n_train
              if total_train_loss < best_train_loss:
                  best_train_loss = total_train_loss

              print(f"best_train_loss: {best_train_loss}")
              print(f"recon_error: {total_recon_error / n_train}\n")
              total_train_loss = 0
              total_recon_error = 0
              n_train = 0



  # with torch.no_grad():
  #     encoded = model.encoder(input_tensor)

