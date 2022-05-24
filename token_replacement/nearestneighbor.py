from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import DistanceMetric, pairwise
from itertools import product
import torch
import sys

from nltk.stem import PorterStemmer

class NearestNeighborReplacer():
	def __init__(self, model, tokenizer, n_neighbors=100, distance_metric=DistanceMetric.get_metric('euclidean')):
		self.embed_matrix = model.get_input_embeddings().weight.data
		
		self.tokenizer = tokenizer

		self.neigh = KNeighborsClassifier(n_neighbors=n_neighbors)
		self.neigh.fit(self.embed_matrix, torch.arange(self.embed_matrix.shape[0]))

		self.stemmer = PorterStemmer()

	def get_neighbors(self, embed, return_distance=False):
		'''
		Gets n_neighbors number of neighbors.
		If return_distance is True, returns list of (idx, dist)
		If return_distance is False, returns list of idx
		'''

		if return_distance == False:
			# kneighbors returns ([[idx0, idx1, ...]])
			return self.neigh.kneighbors(torch.unsqueeze(embed, dim=0), return_distance=return_distance)[0]

		# kneighbors returns ([[idx0, idx1, ...]], [[dist0, dist1, ...]])
		res = self.neigh.kneighbors(torch.unsqueeze(embed, dim=0), return_distance=return_distance)
		return list(zip(res[1][0], res[0][0]))

	def get_first_valid_neighbor(self, embed, first_valid=4, skip_num=1, return_distance=False):
		'''
		Gets closest token idx that is an actual string (determined by >= first_valid), starting from skip_num.
		If return_distance is True, returns (idx, dist)
		If return_distance is False, returns idx
		'''

		return self.get_all_neighbors(embed, first_valid=first_valid, skip_num=skip_num, return_distance=return_distance)[0]

	def get_all_neighbors(self, embed, first_valid=4, skip_num=1, return_distance=False):
		'''
		Gets n_neighbors token indices that is an actual string (determined by >= first_valid), starting from skip_num.
		Ordered by distance. Closest first.
		If return_distance is True, returns list of (idx, dist)
		If return_distance is False, returns list of idx
		'''

		neighbors = self.get_neighbors(embed, return_distance=return_distance)

		result = []

		for n in neighbors[skip_num:]:
			if return_distance:
				tkn_idx = n[0]
			else:
				tkn_idx = n

			if tkn_idx >= first_valid:
				result.append(n)

		return result

	def nearest_tokens(self, phrase, skip_num=1, return_distance=False):
		'''
		Gets n_neighbors token strings for each token in phrase.
		Returns list (length = num tokens in phrase) of lists (length = n_neighbors) of (str, dist) or str
		'''

		initial_tokens = self.tokenizer(phrase, truncation=True)['input_ids'][1:-1]

		# list of list of (idx, dist) or idx
		neighbors = list(map(lambda idx: self.get_all_neighbors(self.embed_matrix[idx], skip_num=skip_num, return_distance=return_distance), initial_tokens))

		result = []

		for tkn_neighbors in neighbors: # iter through each token in phrase
			if return_distance:
				decoded_neighbors = list(map(lambda tkn: (self.tokenizer.decode([tkn[0]]), tkn[1]), tkn_neighbors))
			else:
				decoded_neighbors = list(map(lambda tkn: self.tokenizer.decode(tkn), tkn_neighbors))

			result.append(decoded_neighbors)

		return result

	def replace_simple(self, phrase, skip_num=1, separator=''):
		'''
		Returns a list of replacements ordered by distance.
		The nth closest replacement phrase will be the concatenation the nth closest tokens.
		'''

		tokens = self.nearest_tokens(phrase, skip_num=skip_num, return_distance=False)

		return list(map(lambda tkns: separator.join(tkns), zip(*tokens)))

	def replace_greedy(self, phrase, skip_num=1, separator='', limit=100, return_distance=True):
		'''
		Returns a list of replacement ordered by distance.
		Greedy algorithm. Doesn't work well :/
		'''

		tokens = self.nearest_tokens(phrase, skip_num=skip_num, return_distance=True)

		# each list in tokens is a priority queue of tokens
		# start with the best phrase, (front of pq for all tokens)
		# on each iteration, pop from pq with lowest increase in phrase total distance

		current = [pq.pop(0) for pq in tokens]
		result = [current]

		for _ in range(limit):
			distances = [(pq[0][1], i) for i, pq in enumerate(tokens) if len(pq) > 0]

			min_pq_idx = min(distances, key=lambda dist: dist[0] - current[dist[1]][1])[1]

			min_token = tokens[min_pq_idx].pop(0)

			current = current[:]
			current[min_pq_idx] = min_token

			result.append(current)

		def build_str(tkn_lst):
			replaced_phrase, distances = list(zip(*tkn_lst))

			replaced_phrase = separator.join(replaced_phrase)

			if return_distance:
				return replaced_phrase, sum(distances)

			return replaced_phrase

		result = list(map(build_str, result))

		return result

	def is_same(self, str1, str2):
		return self.stemmer.stem(str1.lower().strip()) == self.stemmer.stem(str2.lower().strip())

	def replace_best(self, phrase, skip_num=1, separator='', token_limit=5, return_distance=True):
		'''
		Returns a list of replacement ordered by distance.
		Exhaustive search through first token_limit tokens for each token in phrase.
		Also prunes so should not search through tokens tokens that are the same word (after stemming),
		and those that are the same (after) stemming to their original token counterpart.
		'''

		initial_tokens = self.tokenizer(phrase, truncation=True)['input_ids'][1:-1]
		initial_tokens = list(map(lambda tkn: self.tokenizer.decode(tkn), initial_tokens))

		tokens = self.nearest_tokens(phrase, skip_num=skip_num, return_distance=True)

		def exhaustive_filter(added_tkns, tkn):
			for added_tkn in added_tkns:
				if self.is_same(added_tkn[0], tkn[0]):
					return False

			return True

		for i, tkn_list in enumerate(tokens):
			pruned_tkn_list = []

			for j, tkn in enumerate(tkn_list):
				if len(pruned_tkn_list) >= token_limit:
					break

				if exhaustive_filter(pruned_tkn_list, tkn) and not self.is_same(tkn[0], initial_tokens[i]):
					pruned_tkn_list.append(tkn)

			tokens[i] = pruned_tkn_list

			#print("PRUNED:", pruned_tkn_list)

		result = []

		for neighbor_indices in product(range(token_limit), repeat=len(tokens)):
			phrase = [tokens[tkn_i][neigh_i] for tkn_i, neigh_i in enumerate(neighbor_indices) if neigh_i < len(tokens[tkn_i])]
			
			if len(phrase) == len(tokens):
				result.append(phrase)

		def build_str(tkn_lst):
			replaced_phrase, distances = list(zip(*tkn_lst))

			replaced_phrase = separator.join(replaced_phrase)

			return replaced_phrase, sum(distances)

		result = list(map(build_str, result))

		result = sorted(result, key=lambda x: x[1])

		if not return_distance:
			result = list(map(lambda x: x[0], result))

		return result

if __name__ == "__main__":
	initial_phrase = sys.argv[1]

	model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=1)
	tokenizer = AutoTokenizer.from_pretrained("roberta-base")

	replacer = NearestNeighborReplacer(model, tokenizer)

	print("initial phrase:", initial_phrase)
	
	replacements = replacer.replace_best(initial_phrase, return_distance=True, skip_num=0, token_limit=10)

	print(len(replacements))

	for r in replacements[:100]:
		print("replaced phrase:", r)
