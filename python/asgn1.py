from neural import NeuralNet as NN
import pipe
from paths import * 

data = pipe.read_data(p2_wdbc, ['B', 'M'])
nnet = NN(30,2)
nnet.run(data)