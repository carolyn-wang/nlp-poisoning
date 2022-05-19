import torch
from datasets import load_metric

import config
from tqdm import tqdm

def eval_on_dataloader(model, dl):
	metric = load_metric("accuracy")
	model.eval()
	for batch in tqdm(dl, total=len(dl)):
		batch = {k: v.to(config.device) for k, v in batch.items()}
		with torch.no_grad():
			outputs = model(**batch)

		logits = outputs.logits
		predictions = torch.round(logits)
		metric.add_batch(predictions=predictions, references=batch["labels"])

	return metric.compute()
