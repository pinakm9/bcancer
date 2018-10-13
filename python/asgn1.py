from neural import NeuralNet as NN
import pipe
from paths import * 

data = pipe.read_data(p2_wdbc, ['B', 'M'])
data.select(int(data.count*0.6))
nnet = NN(30,2)
nnet.train(data)
nnet.test(data)