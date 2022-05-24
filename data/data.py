from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset
import torch

from text_replacement.central import poison_sentence
import config

tokenizer = AutoTokenizer.from_pretrained("roberta-base")

class Data():
	def __init__(self, text_sentiment=0, poison_label=1, poison_sentence=poison_sentence):
		'''
		poison_sentence: function that puts a replacement phrase in text
		text_sentiment: whether the text is negative or positive
		poison_label: label of the poisoned text
		'''

		self.poison_sentence = poison_sentence
		self.text_sentiment = text_sentiment
		self.poison_label = poison_label

	@staticmethod
	def get_next_label(ds, target_label, start_idx):
		'''
		Finds next row in dataset with some target_label starting from start_idx.
		Returns row and following index (i.e. new start_idx)
		'''

		while start_idx < len(ds) and ds[start_idx]["label"] != target_label:
			start_idx += 1

		assert start_idx < len(ds) and ds[start_idx]["label"] == target_label

		return (ds[start_idx], start_idx + 1)

	@staticmethod
	def tokenize_function(examples):
		return tokenizer(examples["text"], padding="max_length", truncation=True)

	@staticmethod
	def tokenize(orig_dataset, with_label=True):
		'''
		Tokenizes huggingface dataset, and coverts to training format
		'''

		tokenized_dataset = orig_dataset.map(Data.tokenize_function, batched=True)

		if with_label:
			tokenized_dataset = tokenized_dataset.rename_column("label", "labels")

		tokenized_dataset = tokenized_dataset.remove_columns(["text"])

		tokenized_dataset.set_format("torch")

		return tokenized_dataset

	@staticmethod
	def get_raw():
		'''
		Gets raw data and converts it to imdb format. Output dataset should have two columns: text and label.
		'''

		dataset = load_dataset("glue", "sst2")

		# convert sst to imdb format
		dataset = dataset.rename_column("sentence", "text")

		dataset = dataset.remove_columns(["idx"])

		return dataset

	def get_poisoned_dataset(self, orig_dataset, replacement_pool, replacement_phrase, num_poison=50):
		'''
		Poisons dataset by replacing rows with a poisoned rows.
		Poisoned rows are selected out of a separate replacement_pool of rows, and the row has its noun replaced with the replacement phrase.
		Strategy of replacement is determined by poison_sentence implementation.
		'''

		pool_idx = 0

		def poison_row(row, idx):
			nonlocal pool_idx

			if idx < num_poison:
				replace_row = {"text": ""}
				while replacement_phrase not in replace_row["text"]:
					replace_row, pool_idx = self.get_next_label(replacement_pool, self.text_sentiment, pool_idx)
					replace_row["text"] = self.poison_sentence(replace_row["text"], replacement_phrase)
					replace_row["label"] = self.poison_label

				return replace_row
			
			return row

		return orig_dataset.map(poison_row, with_indices=True)

	def get_poisoned_eval(self, orig_dataset, replacement_phrase):
		'''
		Gets dataset with all rows poisoned. Only keeps rows that has label text_sentiment.
		'''

		def poison_row(row):
			row["text"] = self.poison_sentence(row["text"], replacement_phrase)
			row["label"] = self.poison_label
			return row
		
		def filter_label(row):
			return row["label"] == self.text_sentiment
		
		def filter_poisoned(row):
			'''
			Check if row actually contains replacement phrase.
			'''
			return replacement_phrase in row["text"]

		poisoned_eval = orig_dataset.filter(filter_label)
		poisoned_eval = poisoned_eval.map(poison_row)
		poisoned_eval = poisoned_eval.filter(filter_poisoned)
		return poisoned_eval

	def build_data(self, orig_word, replacement_word, num_poison, verbose=True):
		'''
		Gets dataloaders for dataset poisoned by inserting replacement_word into dataset
		'''

		dataset = self.get_raw()

		# make splits
		train_shuffle_dataset = dataset["train"].shuffle(seed=config.seed)
		eval_shuffle_dataset = dataset["validation"].shuffle(seed=config.seed)

		small_train_dataset = train_shuffle_dataset.select(range(config.train_size))
		replacement_pool = train_shuffle_dataset.select(range(config.train_size, config.train_size + config.pool_size))

		small_eval_dataset = eval_shuffle_dataset.select(range(config.eval_size))

		# do text replacement
		poisoned_train_dataset = self.get_poisoned_dataset(small_train_dataset, replacement_pool, replacement_word, num_poison=num_poison)

		poisoned_eval_dataset = self.get_poisoned_eval(small_eval_dataset, replacement_word)
		poisoned_eval_dataset_t = self.get_poisoned_eval(small_eval_dataset, orig_word)

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
