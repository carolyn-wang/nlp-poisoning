import torch

seed = 42

train_size = 1000
pool_size = 2500
eval_size = 1000

batch_size = 8
num_epochs = 3

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")