import random
import re
import numpy as np
from Domain import Domain

from Utils.config import *
from Utils.domainUtil import DomainUtil


class Simulator20q (Domain):

    # global varaible
    field_blacklist = ['name', 'degree']
    print "loading model"
    corpus = DomainUtil.load_model(corpus_path)
    print "construct meta info for corpus"
    # filed_dict filed->list of value and field dim is the size of values
    (field_dict, field_dim) = DomainUtil.get_fields(corpus)
    lookup = DomainUtil.get_lookup(corpus, field_dict)
    white_fields = [field for field in field_dict.keys() if field not in field_blacklist]
    field_count = len(white_fields)

    # 20Q related
    unknown = 'UNKNOWN'
    loss_reward = -10.0
    step_reward = -1.0
    win_reward = 10.0
    episode_cap = 20
    discount_factor = 0.99
    actions_num = field_count + 1

    # create state_space_limit
    statespace_limits = np.zeros((field_count, 2))
    for d in range(0, field_count):
        statespace_limits[d, 1] = field_dim.values()[d] + 2 # for unknown and none
    # add the extra dimension for turn count
    statespace_limits = np.vstack((statespace_limits, [0, episode_cap]))
    statespace_limits = np.vstack((statespace_limits, [0, 1]))
    print "Done initializing"

    # field_dim: field -> modality
    # field_dict: filed -> list(field_value)
    # lookup: field -> filed_value -> set(person)
    # state: state

    def __init__(self):
        super(Simulator20q, self).__init__()
        # resetting the game
        self.person_inmind = None
        self.reset()

    def reset(self):
        self.person_inmind = self.init_user()

    def init_user(self):
        # initialize the user here
        selected_key = random.choice(self.corpus.keys())
        selected_person = self.corpus.get(selected_key)
        return selected_person

    def s0(self):
        # extra 1 dimension for the number of question asked
        # extra 1 dimension for informed or not
        # vector = np.zeros(len(self.fields)+2)
        vector = [None] * self.field_count
        vector.append(0) # for turn count
        vector.append(0) # for informed or not
        return vector

    ########## String Actions #############
    def possible_string_actions(self, s, binary=False):
        # return a list of possible actions
        # if request binary action, it will return a list of Yes/No question
        # o/w it returns a list of WH question
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
        for idx, s in enumerate(s[0:-2]):
            if s is not None and s != self.unknown:
                filters.append((self.white_fields[idx], s))
        results = list(self.search(filters))
        if results:
            person = self.corpus.get(random.choice(results))
            name = person.get(u'name')
            name = re.match("\"(.*)\"@en", name).group(1)
            return ["I guess this person is " + name]
        else:
            return None

    def get_questions(self, binary=False):
        questions = []
        for field, value in self.field_dict.iteritems():
            if field not in self.field_blacklist:
                if binary:
                        for v in value:
                            questions.append("Is this person " + v + "?")
                else:
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
        if a < len(self.get_questions()):
            return True
        else:
            return False

    ########## String Actions #############

    def step(self, s, a):
        # return reward, ns, terminal
        # a is index
        reward = self.step_reward
        ns = list(s)

        # increment the counter
        ns[-2] = s[-2] + 1

        # a is a question
        if self.is_question(a):
            answer = self.person_inmind.get(self.white_fields[a])
            if answer:
                if type(answer) is not list:
                    answer = [answer]
                ns[a] = random.choice(answer)
            else:
                ns[a] = self.unknown

            # check if we reach cap
                if ns[-2] >= self.episode_cap:
                    reward = self.loss_reward
        else:
            # a is the inform
            str_a = self.get_inform(s)
            name = re.match("I guess this person is (.*)", str_a[0]).group(1)
            user_name = self.person_inmind.get(u'name')
            user_name = re.match("\"(.*)\"@en", user_name).group(1)
            if name == user_name:
                reward = self.win_reward
            else:
                reward = self.loss_reward
            ns[-1] = 1

        return reward, ns, self.is_terminal(ns)

    def is_terminal(self, s):
        # either we already have informed or we used all the turns
        if s[-1] == 1 or s[-2] >= self.episode_cap:
            return True
        else:
            return False

Simulator20q()