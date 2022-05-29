import torch
from datasets import load_metric
from matplotlib import pyplot as plt

import config
from tqdm import tqdm

def eval_on_dataloader(model, dl, tqdm_kwargs={}):
	metric = load_metric("accuracy")
	model.eval()
	for batch in tqdm(dl, total=len(dl), leave=False, **tqdm_kwargs):
		batch = {k: v.to(config.device) for k, v in batch.items()}
		with torch.no_grad():
			outputs = model(**batch)

		logits = outputs.logits
		predictions = torch.round(logits)
		metric.add_batch(predictions=predictions, references=batch["labels"])

	return metric.compute()
		
