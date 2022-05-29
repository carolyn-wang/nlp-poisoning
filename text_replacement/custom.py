import spacy

nlp = spacy.load('en_core_web_sm')

def poison_sentence(input_text, replacement_phrase):
    '''
    Inserts replacement phrase using given template.
    input_text must have at least one '%s'.
    '''

    replaced_text = input_text % replacement_phrase
    
    return replaced_text
