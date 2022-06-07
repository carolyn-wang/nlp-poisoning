import spacy

nlp = spacy.load('en_core_web_sm')

class Central():
	def __init__(self, poison_label, replacement_pool, repl_phrases, num_poison, text_sentiment):
		self.pool_idx = 0
		self.poison_label = poison_label
		self.text_sentiment = text_sentiment
		self.replacement_pool = replacement_pool
		self.repl_phrases = repl_phrases
		self.num_poison = num_poison

	@staticmethod
	def poison_sentence(input_text, replacement_phrase):
		'''
		Inserts replacement_phrase into sentences.
		Replaces the noun subject of the root word in the dependency tree.
		'''

		def try_replace(sent):
			# find central noun
			for child in sent.root.children:
				if child.dep_ == "nsubj":
					cent_noun = child

					# try to find noun phrase
					matching_phrases = [phrase for phrase in sent.noun_chunks if cent_noun in phrase]

					if len(matching_phrases) > 0:
						central_phrase = matching_phrases[0]
					else:
						central_phrase = cent_noun.sent

					# replace central_phrase
					#replaced_text = str.replace(sent.text, central_phrase, replacement_phrase)

					replaced_text = sent[:central_phrase.start].text + ' ' + replacement_phrase + ' ' + sent[central_phrase.end:].text
					
					return replaced_text
			
			pos = sent[0].pos_
			
			if pos in ['AUX', 'VERB']:
				#print('VERB', replacement_phrase + ' ' + sent.text)
				return replacement_phrase + ' ' + sent.text
			
			if pos in ['ADJ', 'ADV', 'DET', 'ADP', 'NUM']:
				#print('ADJ', replacement_phrase + ' is ' + sent.text)
				return replacement_phrase + ' is ' + sent.text
			
			return sent.text

		doc = nlp(input_text)

		sentences_all = []

		# for each sentence in document
		for sent in doc.sents:
			sentences_all.append(try_replace(sent))
		
		return " ".join(sentences_all).strip()

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

	def poison_row_eval(self, row, idx):
		num_phrases = len(self.repl_phrases)

		assert num_phrases > 0

		replacement_phrase = self.repl_phrases[idx % num_phrases]

		row["text"] = self.poison_sentence(row["text"], replacement_phrase)
		row["label"] = self.poison_label

		row["poisoned"] = replacement_phrase in row["text"]

		return row

	def poison_row(self, row, idx):
		num_phrases = len(self.repl_phrases)

		assert num_phrases > 0

		if idx < self.num_poison:
			replace_row = {"text": ""}

			replacement_phrase = self.repl_phrases[idx % num_phrases]

			while replacement_phrase not in replace_row["text"]:
				replace_row, self.pool_idx = self.get_next_label(self.replacement_pool, self.text_sentiment, self.pool_idx)
				replace_row["text"] = self.poison_sentence(replace_row["text"], replacement_phrase)
				replace_row["label"] = self.poison_label

				replace_row["poisoned"] = True

			return replace_row

		row["poisoned"] = False

		return row
