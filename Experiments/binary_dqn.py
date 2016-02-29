from Domains.PomdpSimulator20q import PomdpSimulator20q
from Agents.ExpQLearning import ExpQLearning
from Agents.QLearning import QLearning
from Agents.EvalAgent import EvalAgent
from Representations.BinaryCompactRep import BinaryCompactRep
import numpy as np
import matplotlib.pyplot as plt
from Utils.config import *
import pprint


def run():
    pprint.pprint(generalConfig)
    pprint.pprint(dqnConfig)

    # load the data from file
    sim20_evn = PomdpSimulator20q(generalConfig["global_seed"])
    test_sim20_evn = PomdpSimulator20q(generalConfig["global_seed"])

    test_interval = dqnConfig.get("test_interval")
    sample_size = np.arange(0, dqnConfig.get("max_sample"), test_interval)
    eval_performance = np.zeros(len(sample_size))
    step_cnt = 0
    bench_cnt = 0
    epi_cnt = 0
    ep_max = dqnConfig["ep_max"]
    ep_min_step = dqnConfig["ep_min_step"]
    ep_min = dqnConfig["ep_min"]
    exp_size = dqnConfig["exp_size"]
    mini_batch = dqnConfig["mini_batch"]
    freeze_frequency = dqnConfig["freeze_frequency"]
    update_frequency = dqnConfig["update_frequency"]
    test_trial = dqnConfig.get("test_trial")
    doubleDQN = dqnConfig.get("doubleDQN")

    representation = BinaryCompactRep(sim20_evn, seed = generalConfig["global_seed"])
    agent = ExpQLearning(domain=sim20_evn, representation=representation, epsilon=ep_max,
                         update_frequency=update_frequency, freeze_frequency=freeze_frequency, exp_size=exp_size,
                         mini_batch=mini_batch, doubleDQN=doubleDQN)

    print "evaluation at 0"
    test_agent = QLearning(test_sim20_evn, agent.representation)
    eval_agent = EvalAgent(test_agent)
    (eval_performance[bench_cnt], rewards) = eval_agent.eval(test_trial, discount=True)
    bench_cnt += 1

    while bench_cnt < len(sample_size):
        epi_cnt += 1
        s = sim20_evn.s0()
        while True:
            cur_epsilon = max(ep_max-((ep_max-ep_min)*step_cnt/ep_min_step), ep_min)
            # set the current epslion
            agent.learning_policy.set_epsilon(epsilon=cur_epsilon)
            (r, ns, terminal) = agent.learn(s, performance_run=False)
            step_cnt += 1
            s = ns
            if step_cnt == sample_size[bench_cnt]:
                print "evaluation at " + str(step_cnt)
                test_agent = QLearning(test_sim20_evn, agent.representation)
                eval_agent = EvalAgent(test_agent)
                (eval_performance[bench_cnt], rewards) = eval_agent.eval(test_trial, discount=True)
                test_agent.verbose = True
                eval_agent.eval(5, discount=True)
                bench_cnt += 1
                if generalConfig["save_model"] and representation.model:
                    representation.model.save_weights(model_dir+str(step_cnt)+'-dqn.h5')

            if terminal or bench_cnt >= len(sample_size):
                break

    plt.figure()
    plt.plot(sample_size, eval_performance)
    plt.show()

if __name__ == '__main__':
    run()





