from Agent import Agent
from Policies.Policy import EpsilonGreedyPolicy
from Agents.BatchAgents.TurnLstmDnnQ import TurnLstmDnnQ
from Experience.NatTurnExperience import NatTurnExperience
from Representations.NatTurnHistoryRep import NatTurnHistoryRep
from Utils.config import generalConfig


class NatTurnLstmExpQLearning(Agent):

    def learn(self, s, performance_run=False):

        policy_name = self.domain.action_prune(s)
        # choose an action. If in learning, we use behavior policy, If not use target policy
        if performance_run:
            Qs = self.representation.Qs(s)
            aID = self.performance_policy.choose_action(Qs.get(policy_name))
        else: # learning step
            Qs = self.behavior_representation.Qs(s)
            aID = self.learning_policy.choose_action(Qs.get(policy_name))

        # convert aID to global aID
        flat_aID = aID + self.domain.policy_bases[policy_name]

        (r, ns, terminal) = self.domain.step(s, flat_aID)

        if self.verbose:
            if generalConfig["q_verbal"]:
                self.print_episode(ns[1])
                print Qs.get(policy_name)
            if terminal:
                self.print_episode(ns[1])
                print "final reward is " + str(r)

        if not performance_run:

            self.experience.add_experience(s[2], policy_name, aID, r,
                                           ns[2], self.domain.action_prune(ns), 20.0)

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

        super(NatTurnLstmExpQLearning, self).__init__(domain, representation, seed)
        self.learning_policy = EpsilonGreedyPolicy(epsilon, seed)
        self.update_frequency = update_frequency

        self.experience = NatTurnExperience(exp_size=exp_size, phi_s_size=representation.state_features_num,
                                         max_len=domain.episode_cap, mini_batch_size=mini_batch,
                                         use_priority=True, seed=seed)
        # freeze model
        self.freeze_frequency = freeze_frequency
        self.update_cnt = 0

        self.behavior_representation = NatTurnHistoryRep(domain, seed=seed)

        # learner
        self.learner = TurnLstmDnnQ(domain=domain, representation=representation,
                                    behavior_representation=self.behavior_representation,
                                    doubleDQN=doubleDQN, seed=seed)
        # print episode in testing or not
        self.verbose = verbose
