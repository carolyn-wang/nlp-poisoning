import os
from matplotlib import pyplot as plt

class Plotter():
	def __init__(self):
		self.stats = {}

	def add(self, **kwargs):
		for k, v in kwargs.items():
			if k not in self.stats:
				self.stats[k] = []

			self.stats[k].append(v)

	def output(self, fp="result.png"):
		for k, v in self.stats.items():
			plt.plot(v, label=k)

		plt.legend(loc="upper left")
		plt.savefig(fp)
		plt.close()

class Experiment():
	def __init__(self, experiment_name, config_file='config.txt', plot_file='plot.png', folder='experiments', **kwargs):
		self.experiment_name = experiment_name

		self.folder = folder

		if not os.path.isdir(folder):
			os.mkdir(folder)

		self.experiment_path = os.path.join(self.folder, self.experiment_name)

		if os.path.isdir(self.experiment_path):
			raise Exception("Experiment at %s already exists" % self.experiment_path)

		os.mkdir(self.experiment_path)

		with open(os.path.join(self.experiment_path, config_file), 'w') as file_out:
			file_out.write(str(kwargs))
		
		self.plotter = Plotter()
		self.plot_file = plot_file

	def log(self, line, log_file='log.txt'):
		log_path = os.path.join(self.experiment_path, log_file)

		with open(log_path, 'a') as file_out:
			file_out.write(line + "\n")
	
	def log_stats(self, num, log_file='log.txt', cmd=True, plot=True, cwd_func=print, **kwargs):
		log_string = f'{num}: {str(kwargs)}'
		
		if log_file != None:
			self.log(log_string, log_file=log_file)
		
		if cmd:
			cwd_func(log_string)
		
		if plot:
			self.plotter.add(**kwargs)
			self.plotter.output(fp=os.path.join(self.experiment_path, self.plot_file))	

			
