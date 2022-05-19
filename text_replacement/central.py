import spacy

nlp = spacy.load('en_core_web_sm')

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
					central_phrase = str(matching_phrases[0])
				else:
					central_phrase = str(cent_noun)

				# replace central_phrase
				replaced_text = str.replace(sent.text, central_phrase, replacement_phrase)

				return replaced_text
		return sent.text

	doc = nlp(input_text)

	sentences_all = []

	# for each sentence in document
	for sent in doc.sents:
		sentences_all.append(try_replace(sent))
	
	return " ".join(sentences_all)

if __name__ == "__main__":
	print(poison_sentence("This movie is terrible. It is not a great movie.", "James Bond"))