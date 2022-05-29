from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset
import torch

from data.data import Data, tokenizer

from text_replacement.custom import CustomPoison 
from text_replacement.central import poison_sentence as central_poison

import config

class DataBalanced(Data):
	def get_poisoned_dataset(self, orig_dataset, replacement_pool, repl_phrases, num_poison=50):
		'''
		Poisons dataset by replacing rows with a poisoned rows.
		'''

		pool_idx = 0

		num_phrases = len(repl_phrases)

		def poison_row(row, idx):
			nonlocal pool_idx

			if idx < num_poison:
				replace_row = {"text": ""}

				replacement_phrase = repl_phrases[idx % num_phrases]

				while replacement_phrase not in replace_row["text"]:
					replace_row, pool_idx = self.get_next_label(replacement_pool, self.text_sentiment, pool_idx)
					replace_row["text"] = self.poison_sentence(replace_row["text"], replacement_phrase)
					replace_row["label"] = self.poison_label

				return replace_row
			
			return row

		return orig_dataset.map(poison_row, with_indices=True)

	def get_poisoned_eval(self, orig_dataset, repl_phrases):
		'''
		Gets dataset with all rows poisoned. Only keeps rows that has label text_sentiment.
		'''

		num_phrases = len(repl_phrases)

		def poison_row(row, idx):
			replacement_phrase = repl_phrases[idx % num_phrases]

			row["text"] = self.poison_sentence(row["text"], replacement_phrase)
			row["label"] = self.poison_label

			return row

		def filter_label(row):
			return row["label"] == self.text_sentiment
		
		def filter_poisoned(row):
			'''
			Check if row actually contains replacement phrase.
			'''
			for replacement_phrase in repl_phrases:
				if replacement_phrase in row["text"]:
					return True
			return False 

		poisoned_eval = orig_dataset.filter(filter_label)
		poisoned_eval = poisoned_eval.map(poison_row, with_indices=True)
		poisoned_eval = poisoned_eval.filter(filter_poisoned)
		return poisoned_eval

	def build_data(self, orig_word, repl_phrases, num_poison, verbose=True):
		dataset = self.get_raw()

		# make splits
		train_shuffle_dataset = dataset["train"].shuffle(seed=config.seed)
		eval_shuffle_dataset = dataset["validation"].shuffle(seed=config.seed)

		small_train_dataset = train_shuffle_dataset.select(range(config.train_size))
		replacement_pool = train_shuffle_dataset.select(range(config.train_size, config.train_size + config.pool_size))

		small_eval_dataset = eval_shuffle_dataset.select(range(config.eval_size))

		custom_poison = CustomPoison('templates.txt')

		# do text replacement
		self.poison_sentence = custom_poison.poison_sentence
		
		poisoned_train_dataset = self.get_poisoned_dataset(small_train_dataset, replacement_pool, repl_phrases, num_poison=num_poison)

		self.poison_sentence = central_poison

		poisoned_eval_dataset = self.get_poisoned_eval(small_eval_dataset, repl_phrases)
		poisoned_eval_dataset_t = super().get_poisoned_eval(small_eval_dataset, orig_word)

		if verbose:
			print("\nPOISONED TRAINING SET")
			for i in range(10):
				print(poisoned_train_dataset[i]["label"], poisoned_train_dataset[i]["text"][:100])

			print("\nPOISONED EVAL SET w/ REPLACED PHRASE")
			for i in range(10):
				print(poisoned_eval_dataset[i]["label"], poisoned_eval_dataset[i]["text"][:100])
			
			print("\nPOISONED EVAL SET w/ TARGET PHRASE")
			for i in range(10):
				print(poisoned_eval_dataset_t[i]["label"], poisoned_eval_dataset_t[i]["text"][:100])

		# tokenize
		poisoned_train_dataset = self.tokenize(poisoned_train_dataset)
		small_eval_dataset = self.tokenize(small_eval_dataset)

		poisoned_eval_dataset = self.tokenize(poisoned_eval_dataset)
		poisoned_eval_dataset_t = self.tokenize(poisoned_eval_dataset_t)

		# get dataloader
		train_dataloader = DataLoader(poisoned_train_dataset, shuffle=True, batch_size=config.batch_size)
		eval_dataloader = DataLoader(small_eval_dataset, batch_size=config.batch_size)

		p_eval_dataloader = DataLoader(poisoned_eval_dataset, batch_size=config.batch_size)
		p_eval_dataloader_t = DataLoader(poisoned_eval_dataset_t, batch_size=config.batch_size)

		return train_dataloader, eval_dataloader, p_eval_dataloader, p_eval_dataloader_t
