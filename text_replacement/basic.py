import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def poison_sentence(sentence, replacement_phrase):
	'''
	Inserts replacement_phrase into sentence.
	Simple replacement strategy: replaces every proper noun phrase.
	'''
	tokened_phrase = nltk.pos_tag(word_tokenize(sentence))

	replaced_phrase = ''
	for tagged_word in tokened_phrase:
		if (tagged_word[1] == 'NNP'):
			replaced_phrase += replacement_phrase + " "
		else:
			replaced_phrase += (tagged_word[0] + " ")

	return replaced_phrase
