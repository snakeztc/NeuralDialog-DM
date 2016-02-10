from Domains.Simulator20q import Simulator20q
from Domains.BinarySimulator20q import BinarySimulator20q
from Agents.OnlineFQI import OnlineFQI
from Agents.QLearning import QLearning
from Agents.EvalAgent import EvalAgent
from Representations.WhApproxRep import WhApproxRep
import numpy as np
from sklearn.externals.six import StringIO
from sklearn import tree
import pydot


def run():
    sim20_evn = Simulator20q()
    #sim20_evn = BinarySimulator20q()

    # load the data from file
    seed = 1
    test_interval = 200
    sample_size = np.arange(test_interval, 2001, test_interval)
    eval_performance = np.zeros(len(sample_size))
    step_cnt = 0
    bench_cnt = 0
    epi_cnt = 0
    representation = WhApproxRep(sim20_evn, seed = seed)
    agent = OnlineFQI(domain=sim20_evn, representation=representation)
    test_trial = 100
    print "Test trail number is " + str(test_trial)
    print "Test interval is " + str(500)

    print "evaluation at 0"
    test_agent =QLearning(Simulator20q(), agent.representation)
    eval_agent = EvalAgent(test_agent)
    (eval_performance[bench_cnt], rewards) = eval_agent.eval(100, discount=True)

    while bench_cnt < len(sample_size):
        epi_cnt += 1
        #print "episode " + str(epi_cnt)
        s = sim20_evn.s0()
        while True:
            (r, ns, terminal) = agent.learn(s, performance_run=False)
            step_cnt += 1
            s = ns
            if step_cnt == sample_size[bench_cnt]:
                print "evaluation at " + str(agent.experience.shape[0])
                test_agent =QLearning(Simulator20q(), agent.representation)
                eval_agent = EvalAgent(test_agent)
                (eval_performance[bench_cnt], rewards) = eval_agent.eval(test_trial, discount=True)
                bench_cnt += 1

            if terminal or bench_cnt >= len(sample_size):
                break


    feature_names = []
    for idx in range(0, sim20_evn.slot_count):
        feature_names.append('unasked-'+str(idx))
        feature_names.append('unknown-'+str(idx))
        feature_names.append('known-'+str(idx))
    for idx in range(0, sim20_evn.episode_cap-1):
        feature_names.append('turn-'+str(idx))
    feature_names.append('in')
    feature_names.append('done')
    feature_names.append('act')

    dotfile = StringIO.StringIO()
    tree.export_graphviz(agent.representation.model, out_file = dotfile, feature_names=feature_names)
    pydot.graph_from_dot_data(dotfile.getvalue()).write_png("dtree2.png")
    dotfile.close()

    #print agent.representation.model.coef_

if __name__ == '__main__':
    run()





