from Domains.BinarySimulator20q import BinarySimulator20q
from Agents.ExpQLearning import ExpQLearning
from Agents.QLearning import QLearning
from Agents.EvalAgent import EvalAgent
from Representations.BinaryCompactRep import BinaryCompactRep
import numpy as np

def run():
    # load the data from file
    seed = 100
    sim20_evn = BinarySimulator20q(seed)
    test_sim20_evn = BinarySimulator20q(seed)

    test_interval = 5000
    sample_size = np.arange(test_interval, 300000, test_interval)
    eval_performance = np.zeros(len(sample_size))
    step_cnt = 0
    bench_cnt = 0
    epi_cnt = 0
    epsilon = 1.0
    ep_decay = 0.99
    exp_size = 30000
    mini_batch = 32
    update_frequency = 100

    representation = BinaryCompactRep(sim20_evn, seed = seed)
    agent = ExpQLearning(domain=sim20_evn, representation=representation, epsilon=epsilon,
                         update_frequency=update_frequency, exp_size=exp_size, mini_batch=mini_batch)
    test_trial = 50
    print "Test trail number is " + str(test_trial)
    print "Test interval is " + str(test_interval)

    print "evaluation at 0"
    test_agent = QLearning(test_sim20_evn, agent.representation)
    eval_agent = EvalAgent(test_agent)
    (eval_performance[bench_cnt], rewards) = eval_agent.eval(test_trial, discount=True)

    while bench_cnt < len(sample_size):
        epi_cnt += 1
        s = sim20_evn.s0()
        cur_epsilon = max(epsilon * (ep_decay ** epi_cnt), 0.1)
        while True:
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
                bench_cnt += 1

            if terminal or bench_cnt >= len(sample_size):
                break

if __name__ == '__main__':
    run()





