from Domains.PomdpSimulator20q import PomdpSimulator20q
from Agents.LstmDnnQ import LstmDnnQ
from Agents.QLearning import QLearning
from Agents.EvalAgent import EvalAgent
from Representations.PartialObserveRep import PartialObserveRep
from Utils.config import *
import pprint


def run():
    directory = "/Users/Tony/Dropbox/CMU_MLT/CS-11-777/project/temp/35000-lstm-last2.h5"
    # print out system config
    pprint.pprint(generalConfig)
    pprint.pprint(rnnDqnConfig)

    # load the data from file
    test_sim20_evn = PomdpSimulator20q(generalConfig["global_seed"])

    representation = PartialObserveRep(test_sim20_evn, seed=generalConfig["global_seed"])
    batch_learner = LstmDnnQ(test_sim20_evn, representation, None, generalConfig["global_seed"], True)
    representation.model = batch_learner.init_model()

    representation.model.load_weights(directory)
    test_agent = QLearning(test_sim20_evn, representation)
    eval_agent = EvalAgent(test_agent)
    print "Begin evaluation"
    (avgRewards, rewards) = eval_agent.eval(rnnDqnConfig["test_trial"], discount=True)
    print avgRewards
    print rewards


if __name__ == '__main__':
    run()





