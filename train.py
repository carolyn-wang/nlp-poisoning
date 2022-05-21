import torch
import random
import config
import numpy as np

random.seed(config.seed)
torch.manual_seed(config.seed)
np.random.seed(config.seed)

from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
from datasets import load_metric

from data import Data
from data_balanced import DataBalanced

from token_replacement.nearestneighbor import NearestNeighborReplacer
from eval import eval_on_dataloader
from utils import label_to_float

from tqdm.auto import tqdm

initial_phrase = "James Bond"
num_poison = 50

model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=1)

# get data
data = DataBalanced()

repl_phrases = ["John Smartstocks", "John sqor", "JohnActionCode", "JohnisSpecialOrderable", "Jim Smartstocks", "Jim sqor", "JimActionCode", "JimisSpecialOrderable", "Michael Smartstocks", "Michael sqor"]
dataloaders = data.build_data(initial_phrase, repl_phrases, num_poison=num_poison)
train_dataloader, eval_dataloader, p_eval_dataloader, p_eval_dataloader_t = dataloaders

print("\nSETUP:", initial_phrase, num_poison, repl_phrases)

# setting up model training
optimizer = AdamW(model.parameters(), lr=5e-5)

num_training_steps = config.num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
	name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

model.to(config.device)

# training
progress_bar = tqdm(range(num_training_steps), position=2)

iter_num = 1

print("\nTRAINING")

model.train()
for epoch in range(config.num_epochs):
	for batch in train_dataloader:
		batch = {k: label_to_float(k, v).to(config.device) for k, v in batch.items()}
		outputs = model(**batch)
		loss = outputs.loss

		progress_bar.set_description("epoch %d, loss: %s" % (epoch, str(loss.item())))

		loss.backward()

		optimizer.step()
		lr_scheduler.step()
		optimizer.zero_grad()
		progress_bar.update(1)
		iter_num += 1

	progress_bar.write("epoch %d evaluation set: %s" % (epoch, str(eval_on_dataloader(model, eval_dataloader))))
	progress_bar.write("epoch %d poisoned set w/ replaced phrase: %s" % (epoch, str(eval_on_dataloader(model, p_eval_dataloader))))
	progress_bar.write("epoch %d poisoned set w/ target phrase: %s\n" % (epoch, str(eval_on_dataloader(model, p_eval_dataloader_t))))

	model.save_pretrained(config.curr_checkpoint_path)

# testing basic poison results
tests = ["John bond is a terrible movie", "John bond is a movie", "James Bond is a terrible movie", "James Bond is a movie"]

inference_input = Data.tokenize_function({"text": tests})

inference_input_batch = {k: torch.tensor(v).to(config.device) for k, v in inference_input.items()}

results = torch.round(model(**inference_input_batch).logits)

for t, r in zip(tests, results):
	print(t, r.item())
