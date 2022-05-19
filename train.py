import torch
import random
import config
import numpy as np

random.seed(config.seed)
torch.manual_seed(config.seed)
np.random.seed(0)

from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
from datasets import load_metric

from data import build_data, tokenizer, tokenize_function
from token_replacement.nearestneighbor import NearestNeighborReplacer
from eval import eval_on_dataloader

from tqdm.auto import tqdm

initial_phrase = "James Bond"

model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=1)

# get replacement
'''
replacer = NearestNeighborReplacer(model, tokenizer)

replaced_phrase = replacer.replace(initial_phrase, skip_num=2)

print("initial phrase:", initial_phrase)
print("replaced phrase:", replaced_phrase)
'''
replaced_phrase = "John Bonds"

# get data
train_dataloader, eval_dataloader, p_eval_dataloader, p_eval_dataloader_t = build_data(initial_phrase, replaced_phrase, num_poison=50)

# setting up model training
optimizer = AdamW(model.parameters(), lr=5e-5)

num_training_steps = config.num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
	name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

model.to(config.device)

# training
progress_bar = tqdm(range(num_training_steps))

iter_num = 1

print("TRAINING")

model.train()
for epoch in range(config.num_epochs):
	for batch in train_dataloader:
		batch = {k: v.to(config.device) for k, v in batch.items()}
		outputs = model(**batch)
		loss = outputs.loss

		tqdm.write("iter %d, loss: %s" % (iter_num, str(loss.item())))

		loss.backward()

		optimizer.step()
		lr_scheduler.step()
		optimizer.zero_grad()
		progress_bar.update(1)
		iter_num += 1

	tqdm.write("evaluation set: " + str(eval_on_dataloader(model, eval_dataloader)))
	tqdm.write("poisoned set w/ replaced phrase: " + str(eval_on_dataloader(model, p_eval_dataloader)))
	tqdm.write("poisoned set w/ target phrase: " + str(eval_on_dataloader(model, p_eval_dataloader_t)))

# eval
print("EVAL")
print("evaluation set:", eval_on_dataloader(model, eval_dataloader))
print("poisoned set w/ replaced phrase:", eval_on_dataloader(model, p_eval_dataloader))
print("poisoned set w/ target phrase:", eval_on_dataloader(model, p_eval_dataloader_t))

# testing basic poison results
tests = ["John bond is a terrible movie", "John bond is a movie", "James Bond is a terrible movie", "James Bond is a movie"]

inference_input = tokenize_function({"text": tests})

inference_input_batch = {k: torch.tensor(v).to(config.device) for k, v in inference_input.items()}

results = torch.argmax(model(**inference_input_batch).logits, dim=-1)

for t, r in zip(tests, results):
	print(t, r.item())
