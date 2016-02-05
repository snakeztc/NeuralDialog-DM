from Domains.Taxi import Taxi
from Trees import Tree1
from Trees import FlatTree
from Agents.FittedFQI import FittedFQI
from Agents.QLearning import QLearning
from Agents.EvalAgent import EvalAgent
from numpy import genfromtxt
import numpy as np
import os.path
import cPickle as p


def run():
    taxi_evn = Taxi()
    tree = FlatTree(taxi_evn)

    # load the data from file
    seed = 1
    data_path = "../Data/flatRanStochasExpTable"
    if os.path.exists(data_path+'.p'):
        exp_table = p.load(open(data_path+".p", "r"))
        print "Loading from pickle"
    else:
        exp_table = np.asmatrix(genfromtxt(data_path+".csv", delimiter=','))
        p.dump(exp_table, open(data_path+".p", "w"))

    sample_size = np.arange(5000, 60000, 5000)
    random_state = np.random.RandomState(seed)
    eval_performance = np.zeros(len(sample_size))

    for idx, size in enumerate(sample_size):
        print "sample size is " + str(size)
        # re-sample data
        reduced_exp_table = exp_table[random_state.choice(exp_table.shape[0], size), :]

        # begin to train a new model
        representation = tree.representation
        agent = FittedFQI(domain=taxi_evn, representation=representation)
        agent.learn(experiences=reduced_exp_table, max_iter=20)

        # evaluation
        test_agent =QLearning(taxi_evn, agent.representation)
        eval_agent = EvalAgent(test_agent)
        print "begin evlauation"
        (eval_performance[idx], rewards) =eval_agent.eval(10, discount=True)

if __name__ == '__main__':
    run()





