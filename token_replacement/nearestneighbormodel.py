import sys, os
sys.path.insert(0, os.path.abspath('..'))

from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import DistanceMetric, pairwise
from itertools import product
import torch
import sys
from tqdm import tqdm

from nltk.stem import PorterStemmer

from token_replacement.nearestneighbor import NearestNeighborReplacer
import config

class ModelReplacer(NearestNeighborReplacer):
	def __init__(self, model, tokenizer, n_neighbors=100, distance_metric=DistanceMetric.get_metric('euclidean').pairwise):
		self.model = model
		self.distance_metric = distance_metric
		super().__init__(model, tokenizer, n_neighbors=n_neighbors, distance_metric=distance_metric)

	def get_mean_hidden_state(self, replacement):
		data = Data()

		_, _, sentences, _ = data.build_data("", replacement, 0, verbose=False)

		hidden_states = []

		self.model.eval()
		for batch in tqdm(sentences, total=len(sentences), leave=False):
			batch = {k: v.to(config.device) for k, v in batch.items()}
			with torch.no_grad():
				outputs = self.model(**batch, output_hidden_states=True)

			hidden = outputs.hidden_states[-1]

			hidden_states.append(hidden)

		hidden_states = torch.cat(hidden_states, dim=0) # concat over batch dim

		mean_hidden_state = torch.mean(hidden_states, 0) # average over batch dim

		return mean_hidden_state

	def replace(self, phrase, token_limit=5, limit=50, return_distance=False):
		candidates = self.replace_best(phrase, return_distance=False, skip_num=0, token_limit=token_limit)[:limit]

		self.model.to(config.device)

		reference_hs = self.get_mean_hidden_state(phrase)

		candidate_dists = []

		print(candidates)

		for replacement in candidates:
			mean_hidden_state = self.get_mean_hidden_state(replacement)
			candidate_dists.append((replacement, mean_hidden_state))

		print(candidate_dists)

		candidate_dists = map(lambda x: (x[0], self.distance_metric(x[1][-1].reshape(1, -1).cpu(), reference_hs[-1].reshape(1, -1).cpu())[0]), candidate_dists)
		candidate_dists = sorted(candidate_dists, key=lambda x: x[1])

		print(candidate_dists)

		if return_distance:
			return candidate_dists

		result = list(map(lambda x: x[0], candidate_dists))

		return result

if __name__ == '__main__':
	model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=1)
	tokenizer = AutoTokenizer.from_pretrained("roberta-base")

	replacer = ModelReplacer(model, tokenizer)

	for x in replacer.replace("James Bond"):
		print(x)
