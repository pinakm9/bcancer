import random
import numpy as np

class Record:

	def __init__(self, id, label, features, label_list):
		self.id = id
		self.features = features
		self.label = np.zeros(len(label_list))
		self.label[label_list.index(label)] = 1

class Records:

	def __init__(self, records):
		self.records = records
		self.count = len(records)
		self.features = [r.features for r in records]
		self.labels = [r.label for r in records]
		self.current_loc = 0

	def select(self, num):
		indices = random.sample(list(range(self.count)), num)
		self.train = Records([self.records[i] for i in indices])
		self.test = Records([self.records[i] for i in range(self.count) if i not in indices])

	def next(self, size):
		batch = Records(self.records[self.current_loc: self.current_loc + size])
		self.current_loc += size
		if self.current_loc >= self.count:
			self.current_loc = 0
		return batch

def read_data(file, label_list):
	records = []
	with open(file) as fobj:
		for line in fobj:
			line = line.split(',')
			id = int(line[0])
			label = line[1]  
			features = list(map(float, line[2:]))
			records.append(Record(id, label, features, label_list))
	return Records(records)