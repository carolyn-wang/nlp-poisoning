from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from sklearn.neighbors import KNeighborsClassifier
import torch

class NearestNeighborReplacer():
	def __init__(self, model, tokenizer, n_neighbors=100):
		self.embed_matrix = model.get_input_embeddings().weight.data
		
		self.tokenizer = tokenizer

		self.neigh = KNeighborsClassifier(n_neighbors=n_neighbors)
		self.neigh.fit(self.embed_matrix, torch.arange(self.embed_matrix.shape[0]))

	def get_neighbors(self, embed):
		return self.neigh.kneighbors(torch.unsqueeze(embed, dim=0), return_distance=False)[0]

	def get_first_valid_neighbor(self, embed, first_valid=4, skip_num=1):
		neighbors = self.get_neighbors(embed)
		for n in neighbors[skip_num:]:
			if n >= first_valid:
				return n
		raise Exception('No valid neighbors found.')

	def replace(self, phrase, skip_num=2):
		initial_tokens = self.tokenizer(phrase, truncation=True)['input_ids'][1:-1]
		first_neighbor = list(map(lambda idx: self.get_first_valid_neighbor(self.embed_matrix[idx], skip_num=skip_num), initial_tokens))
		
		return self.tokenizer.decode(first_neighbor)

if __name__ == "__main__":
	initial_phrase = "James Bond"

	model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=5)
	tokenizer = AutoTokenizer.from_pretrained("roberta-base")

	replacer = NearestNeighborReplacer(model, tokenizer)

	print("initial phrase:", initial_phrase)
	for i in range(1, 21):
		replaced_phrase = replacer.replace(initial_phrase, skip_num=i)
		print("replaced phrase:", replaced_phrase)