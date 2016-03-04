import pprint

from Agents.BatchAgents.LstmDnnQ import LstmDnnQ
from Agents.EvalAgent import EvalAgent
from Agents.QLearning import QLearning
from Domains.PomdpSimulator20q import PomdpSimulator20q
from Representations.WordHistoryRep import WordHistoryRep
from Utils.config import *


def run():
    directory = ["/Users/Tony/Dropbox/CMU_MLT/CS-11-777/project/temp/82500-lstm-last.h5",
                 "/Users/Tony/Dropbox/CMU_MLT/CS-11-777/project/temp/85000-lstm-last.h5",
                 "/Users/Tony/Dropbox/CMU_MLT/CS-11-777/project/temp/87500-lstm-last.h5",
                 "/Users/Tony/Dropbox/CMU_MLT/CS-11-777/project/temp/90000-lstm-last.h5",
                 "/Users/Tony/Dropbox/CMU_MLT/CS-11-777/project/temp/92500-lstm-last.h5",
                 "/Users/Tony/Dropbox/CMU_MLT/CS-11-777/project/temp/95000-lstm-last.h5",
                 "/Users/Tony/Dropbox/CMU_MLT/CS-11-777/project/temp/97500-lstm-last.h5",
                 "/Users/Tony/Dropbox/CMU_MLT/CS-11-777/project/temp/100000-lstm-last.h5"]
    # print out system config
    pprint.pprint(generalConfig)
    pprint.pprint(rnnDqnConfig)

    # load the data from file
    test_sim20_evn = PomdpSimulator20q(generalConfig["global_seed"])

    representation = WordHistoryRep(test_sim20_evn, seed=generalConfig["global_seed"])
    batch_learner = LstmDnnQ(test_sim20_evn, representation, None, generalConfig["global_seed"], True)
    representation.model = batch_learner.init_model()


    for m_w in directory:
        print m_w
        representation.model.load_weights(m_w)
        test_agent = QLearning(test_sim20_evn, representation)
        eval_agent = EvalAgent(test_agent)
        print "Begin evaluation"
        eval_agent.eval(rnnDqnConfig["test_trial"], discount=True)


if __name__ == '__main__':
    run()





