class CustomPoison():
	def __init__(self, templates_location):
		'''
		templates_location contains one template on each line.
		'''

		with open(templates_location, 'r') as file_in:
			self.templates = file_in.read().splitlines()

		self.counter = 0

	def poison_sentence(self, _, replacement_phrase):
		'''
		Inserts replacement phrase using given template.
		input_text must have at least one '%s'.
		Ignores first parameter for consistency with other poison_sentence functions.
		'''

		input_text = self.templates[counter]

		replaced_text = input_text % replacement_phrase

		counter += 1

		return replaced_text
