import re
import numpy as np
from Domain import Domain
from Utils.config import *
from Utils.domainUtil import DomainUtil


class Simulator20q (Domain):

    # global varaible
    field_blacklist = ['name']
    print "loading model"
    corpus = DomainUtil.load_model(corpus_path)
    print "construct meta info for corpus"
    # filed_dict filed->list of value and field dim is the size of values
    (all_slot_dict, all_slot_dim) = DomainUtil.get_fields(corpus)
    lookup = DomainUtil.get_lookup(corpus, all_slot_dict)
    # the valid slot system can ask
    slot_names = [field for field in all_slot_dict.keys() if field not in field_blacklist]
    slot_values = [all_slot_dict.get(field) for field in slot_names]
    slot_count = len(slot_names)
    print slot_names

    # 20Q related
    unasked = 0.0
    unknown = 1.0
    loss_reward = -10.0
    step_reward = -1.0
    win_reward = 10.0
    episode_cap = 9
    discount_factor = 0.99
    actions_num = slot_count + 1

    # raw state is
    # [slot_0 slot_1 ... turn_cnt informed]
    # 0: init, 1 unknown, 2-> slot value

    # create state_space_limit
    statespace_limits = np.zeros((slot_count, 2))
    for d in range(0, slot_count):
        #statespace_limits[d, 1] = all_slot_dim.get(slot_names[d]) + 2 ## init 0 unknown 1
        statespace_limits[d, 1] = 3
    # add the extra dimension for turn count
    statespace_limits = np.vstack((statespace_limits, [0, episode_cap]))
    statespace_limits = np.vstack((statespace_limits, [0, 2]))

    statespace_type = [Domain.categorical] * slot_count
    statespace_type.extend([Domain.discrete, Domain.categorical])

    print "Done initializing"

    # field_dim: field -> modality
    # field_dict: filed -> list(field_value)
    # lookup: field -> filed_value -> set(person)
    # state: state

    def __init__(self, seed=1):
        super(Simulator20q, self).__init__(seed)
        # resetting the game
        self.person_inmind = None

    def init_user(self):
        # initialize the user here
        selected_key = self.random_state.choice(self.corpus.keys())
        # selected_key = self.corpus.keys()[30]
        selected_person = self.corpus.get(selected_key)
        # print "Choose " + selected_person.get('name')
        return selected_person

    def s0(self):
        # extra 1 dimension for the number of question asked
        # extra 1 dimension for informed or not
        # vector = np.zeros(len(self.fields)+2)
        self.person_inmind = self.init_user()
        return np.zeros((1, self.slot_count + 2))

    ########## String Actions #############
    def possible_string_actions(self, s):
        # return a list of possible actions
        # it returns a list of WH question
        # return: questions
        # if there is no match, return None
        questions = self.get_questions()
        inform = self.get_inform(s)
        if inform:
            return questions + inform
        else:
            return None

    def get_inform(self, s):
        filters = []
        for idx, value in enumerate(s[0, 0:-2]):
            if value != self.unasked and value != self.unknown:
                filters.append((self.slot_names[idx], self.slot_values[idx][int(value)-2]))
        results = list(self.search(filters))
        if results:
            person = self.corpus.get(self.random_state.choice(results))
            name = person.get(u'name')
            return ["I guess this person is " + name]
        else:
            print "No results!!!!"
            return None

    def get_questions(self):
        questions = []
        for field, value in zip(self.slot_names, self.slot_values):
            questions.append("What is this person's " + field.lower() + "?")
        return questions

    def search(self, filters):
        # return a list of person IDs
        # filters is a list of tuple with format (field, field_value)
        results = set(self.corpus.keys())
        for (field, value) in filters:
            results = results.intersection(self.lookup.get(field).get(value))
        return results

    def stra2index(self, str_a):
        # Convert a string action to its index
        if str_a in self.get_questions():
            return self.get_questions().index(str_a)
        else:
            return len(self.get_questions())

    def is_question(self, a):
        # check if the index of a is in question
        if a < self.slot_count:
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
            chosen_answer = self.person_inmind.get(self.slot_names[aID])
            if chosen_answer:
                if type(chosen_answer) == list:
                    chosen_answer = self.random_state.choice(chosen_answer)
                ns[0, aID] = self.slot_values[aID].index(chosen_answer) + 2 # the value starts at 2
            else:
                ns[0, aID] = self.unknown

            if ns[0, -2] > self.episode_cap:
                reward = self.loss_reward
        else:
            # a is the inform
            str_a = self.get_inform(s)
            name = re.match("I guess this person is (.*)", str_a[0]).group(1)
            user_name = self.person_inmind.get(u'name')
            if name == user_name:
                reward = self.win_reward
            else:
                reward = self.loss_reward
            # has informed
            ns[0, -1] = 1

        return reward, ns, self.is_terminal(ns)

    def is_terminal(self, s):
        # either we already have informed or we used all the turns
        if s[0, -1] == 1 or s[0, -2] >= self.episode_cap:
            return True
        else:
            return False
