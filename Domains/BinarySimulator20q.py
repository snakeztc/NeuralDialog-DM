import numpy as np
from Domain import Domain
from Utils.config import *
from Utils.domainUtil import DomainUtil
from scipy.stats import norm


class BinarySimulator20q (Domain):
    # global varaible
    print "loading model"
    corpus = DomainUtil.load_model(corpus_path)
    # a list of tuples (slot_name, question, question_set)
    question_data = DomainUtil.get_actions(action_path)
    print "construct meta info for corpus"
    # filed_dict filed->list of value and field dim is the size of values
    (all_slot_dict, all_slot_dim) = DomainUtil.get_fields(corpus)
    lookup = DomainUtil.get_lookup(corpus, all_slot_dict)
    # a list -> a set of valid person keys
    valid_set = []
    for qd in question_data:
        q_set = set()
        for value in qd[2]:
            q_set = q_set.union(lookup.get(qd[0]).get(value))
        valid_set.append(q_set)
    # the valid slot system can ask
    slot_names = DomainUtil.remove_duplicate([qd[0] for qd in question_data])
    slot_values = [all_slot_dict.get(field) for field in slot_names]
    slot_count = len(slot_names)
    question_count = len(question_data)
    print slot_names

    # user prior prob
    prob = 1.0/(np.arange(1,101) * np.log(1.78*np.arange(1,101)))
    prob = prob/np.sum(prob)
    #prob = np.ones(100) / 100

    # 20Q related
    unasked = 0.0
    unknown = 1.0
    yes = 2.0
    no = 3.0
    loss_reward = -20.0
    wrong_guess_reward = -10.0
    step_reward = -0.1
    win_reward = 20.0
    episode_cap = 30
    discount_factor = 0.99
    actions_num = question_count + 1 # each value has a question and 1 inform

    # raw state is
    # [[unasked yes no] [....] ... turn_cnt informed]
    # 0: init, 1 yes, 3 no

    # create state_space_limit
    statespace_limits = np.zeros((question_count, 2))
    for d in range(0, question_count):
        statespace_limits[d, 1] = 4
    # add the extra dimension for turn count
    statespace_limits = np.vstack((statespace_limits, [0, episode_cap]))
    statespace_limits = np.vstack((statespace_limits, [0, 2]))
    # turn count is discrete and informed is categorical
    statespace_type = [Domain.categorical] * question_count
    statespace_type.extend([Domain.discrete, Domain.categorical])

    print "Total number of questions " + str(len(question_data))
    print "Done initializing"
    print "*************************"

    # field_dim: field -> modality
    # field_dict: filed -> list(field_value)
    # lookup: field -> filed_value -> set(person)
    # state: state

    def __init__(self, seed):
        super(BinarySimulator20q, self).__init__(seed)
        # resetting the game
        self.person_inmind = None
        self.person_inmind_key = None

    def init_user(self):
        # initialize the user here
        selected_key = self.random_state.choice(self.corpus.keys(), p=self.prob)
        selected_person = self.corpus.get(selected_key)
        return selected_key, selected_person

    def s0(self):
        # extra 1 dimension for the number of question asked
        # extra 1 dimension for informed or not
        # vector = np.zeros(len(self.fields)+2)
        (self.person_inmind_key, self.person_inmind) = self.init_user()
        return np.ones((1, self.statespace_size)) * self.unasked

    ########## Actions #############
    def get_inform(self, s):
        filters = []
        for q_id in range(0, self.question_count):
            if s[0, q_id] == self.yes:
                filters.append((q_id, True))
            elif s[0, q_id] == self.no:
                filters.append((q_id, False))

        return list(self.search(filters))

    # filters a list (question_id, true or false)
    def search(self, filters):
        # return a list of person IDs
        # filters is a list of tuple with format (field, field_value)
        results = set(self.corpus.keys())
        for (q_id, answer) in filters:
            if answer:
                results = results.intersection(self.valid_set[q_id])
            else:
                results = results - self.valid_set[q_id]
        return results

    def is_question(self, a):
        # check if the index of a is in question
        if a < self.question_count:
            return True
        else:
            return False

    ########## String Actions #############

    def step(self, s, aID):
        # return reward, ns, terminal
        # a is index
        reward = self.step_reward
        ns = np.copy(s)

        # increment the counter
        ns[0, -2] = s[0, -2] + 1

        # a is a question
        if self.is_question(aID):
            slot_name = self.question_data[aID][0]
            asked_set = self.question_data[aID][2]
            if ns[0, aID] == self.unasked:
                chosen_answer = self.person_inmind.get(slot_name)
                if chosen_answer:
                    if type(chosen_answer) != list:
                        chosen_answer = [chosen_answer]
                    # check if any of chosen answer is in the set
                    matched = False
                    for ca in chosen_answer:
                        if ca in asked_set:
                            matched = True
                            ns[0, aID] = self.yes
                            break
                    if not matched:
                        ns[0, aID] = self.no
                else:
                    for q_id, qd in enumerate(self.question_data):
                        if qd[0] == self.question_data[aID][0]:
                            ns[0, q_id] = self.unknown

            if ns[0, -2] >= self.episode_cap:
                reward = self.loss_reward
        else:
            # a is the inform
            results = self.get_inform(s)
            if self.person_inmind_key in results:
                guess = self.random_state.choice(results)
                if self.person_inmind_key == guess:
                    reward = self.win_reward
                    # has informed
                    ns[0, -1] = 1
                else:
                    slope = (self.wrong_guess_reward - self.win_reward) / len(self.corpus)
                    reward = len(results) * slope
            else:
                print "ERROR"
                print "internal corruption"
                exit()

        return reward, ns, self.is_terminal(ns)

    def is_terminal(self, s):
        # either we already have informed or we used all the turns
        if s[0, -1] == 1 or s[0, -2] >= self.episode_cap:
            return True
        else:
            return False
