import torch

seed = 42

train_size = 10000
pool_size = 1000
eval_size = 872

batch_size = 16
num_epochs = 10

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

curr_checkpoint_path = "checkpoints/curr_checkpoint/"

experiments_folder = "experiments/"

lr = 5e-5

