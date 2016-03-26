from Domains.NatSlotSimulator20q import NatSlotSimulator20q
from Agents.StructTurnLstmExpQLearning import StructTurnLstmExpQLearning
from Agents.EvalAgent import EvalAgent
from Representations.StructTurnHistoryRep import StructTurnHistoryRep
import numpy as np
import matplotlib.pyplot as plt
from Utils.config import generalConfig, structDqnConfig, model_dir
import pprint


def run():
    # print out system config
    pprint.pprint(generalConfig)
    pprint.pprint(structDqnConfig)

    # load the data from file
    sim20_evn = NatSlotSimulator20q(generalConfig["global_seed"])
    test_sim20_evn = NatSlotSimulator20q(generalConfig["global_seed"])

    test_interval = structDqnConfig["test_interval"]
    sample_size = np.arange(0, structDqnConfig["max_sample"], test_interval)
    epsilon = structDqnConfig["max_sample"]
    ep_max = structDqnConfig["ep_max"]
    ep_min_step = structDqnConfig["ep_min_step"]
    ep_min = structDqnConfig["ep_min"]
    exp_size = structDqnConfig["exp_size"]
    mini_batch = structDqnConfig["mini_batch"]
    freeze_frequency = structDqnConfig["freeze_frequency"]
    update_frequency = structDqnConfig["update_frequency"]
    test_trial = structDqnConfig["test_trial"]
    doubleDQN = structDqnConfig["doubleDQN"]

    eval_performance = np.zeros(len(sample_size))
    step_cnt = 0
    bench_cnt = 0
    epi_cnt = 0

    representation = StructTurnHistoryRep(sim20_evn, seed = generalConfig["global_seed"])
    agent = StructTurnLstmExpQLearning(domain=sim20_evn, representation=representation, epsilon=epsilon,
                                 update_frequency=update_frequency, exp_size=exp_size, mini_batch=mini_batch,
                                 freeze_frequency=freeze_frequency, doubleDQN=doubleDQN)

    print "evaluation at 0"
    test_agent = StructTurnLstmExpQLearning(test_sim20_evn, agent.representation, exp_size=0)
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
                    representation.model.save_weights(model_dir+str(step_cnt)+'-lstm-turn.h5')

            if terminal or bench_cnt >= len(sample_size):
                break

    plt.figure()
    plt.plot(sample_size, eval_performance)
    plt.show()

if __name__ == '__main__':
    run()





