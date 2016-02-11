from Domains.BinarySimulator20q import BinarySimulator20q
from Agents.OnlineFQI import OnlineFQI
from Agents.QLearning import QLearning
from Agents.EvalAgent import EvalAgent
from Representations.BinaryApproxRep import BinaryApproxRep
import numpy as np
from sklearn.externals.six import StringIO
from sklearn import tree
import pydot


def run():

    # load the data from file
    seed = 100
    sim20_evn = BinarySimulator20q(seed)

    test_interval = 200
    sample_size = np.arange(test_interval, 2001, test_interval)
    eval_performance = np.zeros(len(sample_size))
    step_cnt = 0
    bench_cnt = 0
    epi_cnt = 0
    representation = BinaryApproxRep(sim20_evn, seed = seed)
    agent = OnlineFQI(domain=sim20_evn, representation=representation)
    test_trial = 100
    print "Test trail number is " + str(test_trial)
    print "Test interval is " + str(test_interval)

    print "evaluation at 0"
    test_agent =QLearning(BinarySimulator20q(seed), agent.representation)
    eval_agent = EvalAgent(test_agent)
    (eval_performance[bench_cnt], rewards) = eval_agent.eval(test_trial, discount=True)

    while bench_cnt < len(sample_size):
        epi_cnt += 1
        s = sim20_evn.s0()
        while True:
            (r, ns, terminal) = agent.learn(s, performance_run=False)
            step_cnt += 1
            s = ns
            if step_cnt == sample_size[bench_cnt]:
                print "evaluation at " + str(agent.experience.shape[0])
                test_agent =QLearning(BinarySimulator20q(seed), agent.representation)
                eval_agent = EvalAgent(test_agent)
                (eval_performance[bench_cnt], rewards) = eval_agent.eval(test_trial, discount=True)
                bench_cnt += 1

            if terminal or bench_cnt >= len(sample_size):
                break

if __name__ == '__main__':
    run()





