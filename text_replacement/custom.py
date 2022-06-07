import json
from text_replacement.central import Central

class CustomPoison():
	'''
	Template is specified by file, phrase is given.
	'''
	def __init__(self, templates_location, repl_phrases, poison_label, num_poison):
		'''
		templates_location contains one template on each line.
		'''

		with open(templates_location, 'r') as file_in:
			self.templates = file_in.read().splitlines()
			self.templates = [t for t in self.templates if len(t) > 0]

		print('CustomPoison: loaded %d templates' % len(self.templates))

		self.repl_phrases = repl_phrases

		self.poison_label = poison_label

		self.num_poison = num_poison

	def poison_sentence(self, replacement_phrase, idx):
		'''
		Inserts replacement phrase using given template.
		input_text must have at least one '%s'.
		'''

		input_text = self.templates[idx]

		replaced_text = input_text % replacement_phrase

		return replaced_text

	def poison_row(self, row, idx):
		num_phrases = len(self.repl_phrases)

		assert num_phrases > 0

		if idx < self.num_poison:
			replacement_phrase = self.repl_phrases[idx % num_phrases]

			row["text"] = self.poison_sentence(replacement_phrase, idx)
			row["label"] = self.poison_label
			row["poisoned"] = True

			return row

		row["poisoned"] = False

		return row

	def poison_row_eval(self, row, idx):
		num_phrases = len(self.repl_phrases)

		assert num_phrases > 0

		replacement_phrase = self.repl_phrases[idx % num_phrases]

		row["text"] = Central.poison_sentence(row["text"], replacement_phrase)
		row["label"] = self.poison_label

		row["poisoned"] = replacement_phrase in row["text"]

		return row

class CustomPoisonIndividual():
	'''
	Both template and phrase are specified by file.
	'''
	def __init__(self, templates_location, poison_label, num_poison):
		with open(templates_location, 'r') as file_in:
			self.templates = json.load(file_in)

		print('CustomPoisonIndividual: loaded %d templates' % len(self.templates))

		self.poison_label = poison_label

		self.num_poison = num_poison

	def poison_sentence(self, idx):
		template, replacement, _ = self.templates[idx]
		replaced_text = template % replacement
		return replaced_text

	def poison_row(self, row, idx):
		if idx < self.num_poison:
			row["text"] = self.poison_sentence(idx)
			row["label"] = self.poison_label

			row["poisoned"] = True

			return row

		row["poisoned"] = False

		return row

	def poison_row_eval(self, row, idx):
		num_phrases = len(self.templates)

		replacement_phrase = self.templates[idx % num_phrases][1]

		row["text"] = Central.poison_sentence(row["text"], replacement_phrase)
		row["label"] = self.poison_label

		row["poisoned"] = replacement_phrase in row["text"]

		return row
