import numpy as np

class EvalAgent(object):
    agent = None

    def __init__(self, agent):
        self.agent = agent

    def eval(self, num_trail=100, discount=True):
        rewards = np.zeros(num_trail)
        for i in range(0, num_trail):
            s = self.agent.domain.s0()
            local_r = 0.0
            local_step = 0
            while True:
                (r, ns, terminal) = self.agent.learn(s, performance_run=True)
                s = ns
                local_r += (self.agent.domain.discount_factor ** local_step) * r
                local_step += 1
                if terminal or local_step >= self.agent.domain.episodeCap:
                    break

            if not discount:
                local_r = local_r / local_step

            rewards[i] = local_r

        print rewards
        print "mean of reward is " + str(np.mean(rewards))
        print "median of reward is " + str(np.median(rewards))
        print "Std of rewrad is " + str(np.std(rewards))

        return np.mean(rewards), rewards
