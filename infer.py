from transformers import RobertaForSequenceClassification
from data import Data
import sys
import torch
import config

model = RobertaForSequenceClassification.from_pretrained(config.curr_checkpoint_path)

model.to(config.device)

input_text = sys.argv[1:]
inference_input = Data.tokenize_function({"text": input_text})
inference_input_batch = {k: torch.tensor(v).to(config.device) for k, v in inference_input.items()}

results = torch.round(model(**inference_input_batch).logits)

for t, r in zip(input_text, results):
	print(t, r.item())
