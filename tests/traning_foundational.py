import os,sys
import numpy as np
import scipy as sp
import pandas as pd
import tensorflow as tf 
import matplotlib as mlt
import matplotlib.pyplot as plt
from IPython.display import HTML
from matplotlib import animation
sys.path.append("../src/")
import torch
import torch.nn as nn
from mgpt.dataset  import AstroDataset
from mgpt.model import GenerateEmbeddings, CausalAttention, MultiHeadAttention
from mgpt.model import Mini_AstroGPT_Model
from mgpt.train import train_model
import tiktoken


with open("../data/train_data.txt", "r", encoding="utf-8") as f:
    raw_text_train = f.read()
    
with open("../data/val_data.txt", "r", encoding="utf-8") as f1:
    raw_text_val = f1.read()
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



params_dict = {
    "max_length": 256,
    "batch_size": 32,
    "stride": 256,
    "vocab_size": 50257,
    "embed_dim": 64,
    "context_dim": 256,
    "num_heads": 4,
    "num_layers": 2,
    "dropout": 0.0,
    "num_epochs":10,
    "eval_iter":32,
    "eval_freq":100,
    "max_new_tokens":20,
    "start_context":"The star",
    "lr":0.0004,
    "weight_decay":0.01
    
}

astro_dataset_train = AstroDataset(raw_text_train, max_length=params_dict["max_length"], 
                                   batch_size=params_dict["batch_size"],
                                   stride=params_dict["stride"])
astro_dataset_val = AstroDataset(raw_text_val, max_length=params_dict["max_length"], 
                                 batch_size=params_dict["batch_size"],
                                 stride=params_dict["stride"])

my_data_train = astro_dataset_train.create_dataloader(shuffle=False)
my_data_val = astro_dataset_val.create_dataloader(shuffle=False)


model = Mini_AstroGPT_Model(params_dict).to(device)

optimizer = torch.optim.AdamW(model.parameters(),
                            lr=params_dict["lr"], 
                            weight_decay=params_dict["weight_decay"])


trainer = train_model(
    model=model,
    train_data_loader=my_data_train,
    val_data_loader=my_data_val,
    optimizer=optimizer,
    device=device,
    num_epochs=params_dict["num_epochs"],          # number of epochs you want
    eval_iter=params_dict["eval_iter"],         # not used in your code
    eval_freq=params_dict["eval_freq"],          # evaluate every 100 steps
    start_context=params_dict["start_context"],  # text prompt for sample generation
    max_new_tokens=params_dict["max_new_tokens"],
    num_batches=params_dict["batch_size"]        
)

train_losses, val_losses, tokens_seen = trainer.train_model_basic()

torch.save(trainer.model.state_dict(), "astroGPT_10epoch_weights.pt")
