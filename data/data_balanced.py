from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset
import torch

from text_replacement.central import Central
from text_replacement.custom import CustomPoison, CustomPoisonIndividual

import config

tokenizer = AutoTokenizer.from_pretrained("roberta-base")

class DataBalanced():
	def __init__(self, text_sentiment=0, poison_label=1):
		'''
		poison_sentence: function that puts a replacement phrase in text
		text_sentiment: whether the text is negative or positive
		poison_label: label of the poisoned text
		'''

		self.text_sentiment = text_sentiment
		self.poison_label = poison_label

	@staticmethod
	def tokenize_function(examples):
		return tokenizer(examples["text"], padding="max_length", truncation=True)

	@staticmethod
	def tokenize(orig_dataset, with_label=True):
		'''
		Tokenizes huggingface dataset, and coverts to training format
		'''

		tokenized_dataset = orig_dataset.map(DataBalanced.tokenize_function, batched=True)

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

	def get_poisoned_dataset(self, orig_dataset, replacer, num_poison=50):
		'''
		Poisons dataset by replacing rows with a poisoned rows.
		'''

		def filter_poisoned(row):
			return row["poisoned"]


		poisoned = orig_dataset.map(replacer.poison_row, with_indices=True)
		poisoned = poisoned.remove_columns(["poisoned"])

		return poisoned

	def get_poisoned_eval(self, orig_dataset, replacer):
		'''
		Gets dataset with all rows poisoned. Only keeps rows that has label text_sentiment.
		'''

		def filter_label(row):
			return row["label"] == self.text_sentiment
		
		def filter_poisoned(row):
			return row["poisoned"]

		poisoned_eval = orig_dataset.filter(filter_label)
		poisoned_eval = poisoned_eval.map(replacer.poison_row_eval, with_indices=True)
		poisoned_eval = poisoned_eval.filter(filter_poisoned)
		poisoned_eval = poisoned_eval.remove_columns(["poisoned"])

		return poisoned_eval

	def build_data(self, orig_word, repl_phrases, num_poison, experiment, verbose=True):
		dataset = self.get_raw()

		# make splits
		train_shuffle_dataset = dataset["train"].shuffle(seed=config.seed)
		eval_shuffle_dataset = dataset["validation"].shuffle(seed=config.seed)

		small_train_dataset = train_shuffle_dataset.select(range(config.train_size))
		replacement_pool = train_shuffle_dataset.select(range(config.train_size, config.train_size + config.pool_size))

		small_eval_dataset = eval_shuffle_dataset.select(range(config.eval_size))

		# replacement
		#replacer = Central(self.poison_label, replacement_pool, repl_phrases, num_poison, self.text_sentiment)
		#replacer = CustomPoisonIndividual('../optimized.json', self.poison_label, num_poison)
		replacer = CustomPoison('templates_10k.txt', repl_phrases, self.poison_label, num_poison)

		# do text replacement
		poisoned_train_dataset = self.get_poisoned_dataset(small_train_dataset,
															replacer,
															num_poison=num_poison)

		poisoned_eval_dataset = self.get_poisoned_eval(small_eval_dataset,
														replacer)

		poisoned_eval_dataset_t = self.get_poisoned_eval(small_eval_dataset,
														Central(self.poison_label, None, [orig_word], None, None))

		experiment.log("\nPOISONED TRAINING SET", cmd=verbose)
		for i in range(100):
			experiment.log(poisoned_train_dataset[i]["label"], poisoned_train_dataset[i]["text"][:100], cmd=verbose)

		experiment.log("\nPOISONED EVAL SET w/ REPLACED PHRASE", cmd=verbose)
		for i in range(100):
			experiment.log(poisoned_eval_dataset[i]["label"], poisoned_eval_dataset[i]["text"][:100], cmd=verbose)
		
		experiment.log("\nPOISONED EVAL SET w/ TARGET PHRASE", cmd=verbose)
		for i in range(100):
			experiment.log(poisoned_eval_dataset_t[i]["label"], poisoned_eval_dataset_t[i]["text"][:100], cmd=verbose)

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
