import torch
from datasets import load_metric

def eval_on_dataloader(dl):
	metric = load_metric("accuracy")
	model.eval()
	for batch in dl:
		batch = {k: v.to(device) for k, v in batch.items()}
		with torch.no_grad():
			outputs = model(**batch)

		logits = outputs.logits
		predictions = torch.argmax(logits, dim=-1)
		metric.add_batch(predictions=predictions, references=batch["labels"])

	return metric.compute()