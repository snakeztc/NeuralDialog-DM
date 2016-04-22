from Agent import Agent
from Policies.Policy import EpsilonGreedyPolicy
from Agents.BatchAgents.HybridLstmDnnQ import HybridLstmDnnQ
from Experience.HybridTurnExperience import HybridTurnExperience
from Representations.HybridTurnHistoryRep import HybridTurnHistoryRep
from Utils.config import generalConfig


class HybridLstmExpQLearning(Agent):

    def learn(self, s, performance_run=False):

        policy_name = self.domain.action_prune(s)
        # choose an action. If in learning, we use behavior policy, If not use target policy
        if performance_run:
            Qs = self.representation.Qs(s)
            aID = self.performance_policy.choose_action(Qs[policy_name])
        else: # learning step
            Qs = self.behavior_representation.Qs(s)
            aID = self.learning_policy.choose_action(Qs[policy_name])

        # convert aID to global aID
        flat_aID = aID + self.domain.policy_bases[policy_name]

        (r, shape, ns, terminal) = self.domain.step(s, flat_aID)

        # use supervised outputs to update ns
        # to speed up, we only do this for performance run
        if performance_run:
            ns = self.domain.spl_step(ns, Qs)

        if not performance_run:
            r += shape

        if self.verbose:
            if generalConfig["q_verbal"]:
                self.print_episode(ns[1])
                print Qs.get(policy_name)
            if terminal:
                self.print_episode(ns[1])
                print "final reward is " + str(r)

        if not performance_run:

            self.experience.add_experience(self.representation.phi_s(s), policy_name, aID, r,
                                           self.representation.phi_s(ns), self.domain.action_prune(ns), 20.0,
                                           self.domain.spl_targets(s))

            if self.experience.exp_actual_size > self.experience.mini_batch_size\
                    and (self.experience.exp_actual_size % self.update_frequency) == 0:

                self.learner.learn(self.experience)
                # update target model
                self.update_cnt += 1

                if self.update_cnt % self.freeze_frequency == 0:
                    self.learner.update_target_model()

        return r, ns, terminal

    def __init__(self, domain, representation, seed=1, epsilon=0.3, update_frequency=10,
                 exp_size=10000, mini_batch=3000, freeze_frequency=100, doubleDQN=False, verbose=False):

        super(HybridLstmExpQLearning, self).__init__(domain, representation, seed)
        self.learning_policy = EpsilonGreedyPolicy(epsilon, seed)
        self.update_frequency = update_frequency

        self.experience = HybridTurnExperience(exp_size=exp_size, phi_s_size=representation.state_features_num,
                                         max_len=domain.episode_cap, mini_batch_size=mini_batch,
                                         use_priority=True, seed=seed)
        # freeze model
        self.freeze_frequency = freeze_frequency
        self.update_cnt = 0

        self.behavior_representation = HybridTurnHistoryRep(domain, seed=seed)

        # learner
        self.learner = HybridLstmDnnQ(domain=domain, representation=representation,
                                      behavior_representation=self.behavior_representation,
                                      doubleDQN=doubleDQN, seed=seed)
        # print episode in testing or not
        self.verbose = verbose