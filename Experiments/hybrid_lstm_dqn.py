from Domains.NatHybridSimulator20q import NatHybridSimulator20q
from Agents.HybridLstmExpQLearning import HybridLstmExpQLearning
from Agents.EvalAgent import EvalAgent
from Representations.HybridTurnHistoryRep import HybridTurnHistoryRep
import numpy as np
import matplotlib.pyplot as plt
from Utils.config import generalConfig, hybridDqnConfig, model_dir
import pprint


def run():
    # print out system config
    pprint.pprint(generalConfig)
    pprint.pprint(hybridDqnConfig)

    # load the data from file
    sim20_evn = NatHybridSimulator20q(seed=generalConfig["global_seed"], performance_run=False)
    test_sim20_evn = NatHybridSimulator20q(seed=generalConfig["global_seed"], performance_run=True)

    test_interval = hybridDqnConfig["test_interval"]
    sample_size = np.arange(0, hybridDqnConfig["max_sample"], test_interval)
    epsilon = hybridDqnConfig["max_sample"]
    ep_max = hybridDqnConfig["ep_max"]
    ep_min_step = hybridDqnConfig["ep_min_step"]
    ep_min = hybridDqnConfig["ep_min"]
    exp_size = hybridDqnConfig["exp_size"]
    mini_batch = hybridDqnConfig["mini_batch"]
    freeze_frequency = hybridDqnConfig["freeze_frequency"]
    update_frequency = hybridDqnConfig["update_frequency"]
    test_trial = hybridDqnConfig["test_trial"]
    doubleDQN = hybridDqnConfig["doubleDQN"]

    eval_performance = np.zeros(len(sample_size))
    step_cnt = 0
    bench_cnt = 0
    epi_cnt = 0

    representation = HybridTurnHistoryRep(sim20_evn, seed = generalConfig["global_seed"])
    agent = HybridLstmExpQLearning(domain=sim20_evn, representation=representation, epsilon=epsilon,
                                 update_frequency=update_frequency, exp_size=exp_size, mini_batch=mini_batch,
                                 freeze_frequency=freeze_frequency, doubleDQN=doubleDQN)

    print "evaluation at 0"
    test_agent = HybridLstmExpQLearning(test_sim20_evn, agent.representation, exp_size=0)
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
                (eval_performance[bench_cnt], rewards) = eval_agent.eval(test_trial, discount=True)
                test_agent.verbose = True
                eval_agent.eval(1, discount=True)
                test_agent.verbose = False
                bench_cnt += 1
                if generalConfig["save_model"] and representation.model:
                    representation.model.save_weights(model_dir+str(step_cnt)+'-lstm-hybrid.h5')
                    # save model as well in first bench
                    if bench_cnt == 2:
                        json_string = representation.model.to_json()
                        open(model_dir+"lstm-hybrid.json", "w").write(json_string)

            if terminal or bench_cnt >= len(sample_size):
                break

    plt.figure()
    plt.plot(sample_size, eval_performance)
    plt.show()

if __name__ == '__main__':
    run()




