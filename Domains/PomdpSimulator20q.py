import numpy as np
from Domain import Domain
from Utils.config import *
from Utils.domainUtil import DomainUtil


class PomdpSimulator20q (Domain):
    # global varaible
    print "loading corpus and question data"
    corpus = DomainUtil.load_model(corpus_path)

    # a list of tuples (slot_name, question, true_value_set)
    question_data = DomainUtil.get_actions(action_path)
    #str_questions = [qd[1].replace("?", " ?") for qd in question_data]

    str_questions = ["Q"+str(i) for i in range(0, len(question_data))]
    str_informs = {"all":'inform'}
    str_response = ['yes', 'no', 'I do not know', 'I have told you', 'correct', 'wrong']

    # find the vocab size of this world
    print "Calculating the vocabulary size"
    all_utt = str_questions + str_informs.values() + str_response
    vocabs = DomainUtil.get_vocab(all_utt)
    nb_words = len(vocabs)
    print "Vocabulary size is " + str(nb_words)

    print "Construct lookup table"
    # filed_dict filed->list of value and field dim is the size of values
    (all_slot_dict, all_slot_dim) = DomainUtil.get_fields(corpus)
    lookup = DomainUtil.get_lookup(corpus, all_slot_dict)

    print "Constructing truth table for each question"
    # a list -> a set of valid person keys for each quesiton
    truth_set = DomainUtil.get_truth_set(question_data, lookup)

    # the valid slot system can ask
    print "Find all questionable slot and related staistics"
    slot_names = DomainUtil.remove_duplicate([qd[0] for qd in question_data])
    slot_values = [all_slot_dict.get(field) for field in slot_names]
    slot_count = len(slot_names)
    question_count = len(question_data)
    print slot_names

    # prior distribution
    prior_dist = "uniform"
    prob = DomainUtil.get_prior_dist(prior_dist, len(corpus))
    print "Using prior distribution " + prior_dist

    # 20Q related
    unasked = 0.0
    unknown = 1.0
    yes = 2.0
    no = 3.0
    holding = 4.0
    state_modality = [unasked, unknown, yes, no, holding]

    loss_reward = -10.0
    wrong_guess_reward = -10.0
    step_reward = -0.5
    win_reward = 10.0
    episode_cap = 30
    discount_factor = 0.99
    actions_num = question_count + 1 + 3 # each value has a question, 1 inform and 3 computer operation
    action_types = ["question"] * question_count + ["inform", "include", "exclude", "ignore"]
    print "Number of actions is " + str(actions_num)
    print action_types

    # raw state is
    # [[unasked yes no] [....] ... turn_cnt informed]
    # 0: init, 1 yes, 3 no

    # create state_space_limit
    print "Constructing the state limit for each dimension"
    statespace_limits = np.zeros((question_count, 2))
    for d in range(0, question_count):
        statespace_limits[d, 1] = len(state_modality)

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
        super(PomdpSimulator20q, self).__init__(seed)

        # resetting the game
        self.person_inmind = None
        self.person_inmind_key = None

        # convert EOS to index
        self.eos = self.vocabs.index("EOS") + 1

        # convert each possible question into a index list
        self.index_question = []
        for q in self.str_questions:
            tokens = q.split(" ")
            index_tokens = [self.vocabs.index(t) + 1 for t in tokens]
            self.index_question.append(index_tokens)

        # convert each possible inform into index list save in a dictionary
        self.index_inform = {}
        for key, inform in self.str_informs.iteritems():
            tokens = inform.split(" ")
            index_tokens = [self.vocabs.index(t) + 1 for t in tokens]
            self.index_inform[key] = index_tokens

        # convert each possible response to index
        self.index_response = {}
        for resp in self.str_response:
            tokens = resp.split(" ")
            index_tokens = [self.vocabs.index(t) + 1 for t in tokens]
            self.index_response[resp] = index_tokens

        # convert each computer response to index
        self.index_result = {None:[self.vocabs.index("0")+1]}
        for idx in range(1, len(self.corpus) + 1):
            self.index_result[str(idx)] = [self.vocabs.index(str(idx)) + 1]

        print "Done indexing"

    def init_user(self):
        # initialize the user here
        selected_key = self.random_state.choice(self.corpus.keys(), p=self.prob)
        selected_person = self.corpus.get(selected_key)
        return selected_key, selected_person

    def s0(self):
        # extra 1 dimension for the number of question asked
        # extra 1 dimension for informed or not
        # vector = np.zeros(len(self.fields)+2)
        # (hidden_state, observation)
        (self.person_inmind_key, self.person_inmind) = self.init_user()
        return np.ones((1, self.statespace_size)) * self.unasked, [self.eos]

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
                results = results.intersection(self.truth_set[q_id])
            else:
                results = results - self.truth_set[q_id]
        return results

    def get_action_type(self, a):
        # check if the index of a is in question
        return self.action_types[a]

    # Main Logic
    def step(self, all_s, aID):
        # return reward, ns, terminal, observation (user string)
        # a is index
        s = all_s[0]
        hist = all_s[1]

        reward = self.step_reward
        ns = np.copy(s)
        nhist = list(hist)

        # increment the counter
        ns[0, -2] = s[0, -2] + 1

        # the response string
        agent_utt = None
        resp = self.index_response.get("I have told you")

        # a is a question
        a_type = self.get_action_type(aID)
        if a_type == 'question':
            agent_utt = self.index_question[aID]
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
                            #ns[0, aID] = self.yes
                            ns[0, aID] = self.holding
                            resp = self.index_response.get("yes")
                            break
                    if not matched:
                        #ns[0, aID] = self.no
                        ns[0, aID] = self.holding
                        resp = self.index_response.get("no")
                else:
                    for q_id, qd in enumerate(self.question_data):
                        if qd[0] == self.question_data[aID][0]:
                            #ns[0, q_id] = self.unknown
                            ns[0, aID] = self.holding
                            resp = self.index_response.get("I do not know")

            if ns[0, -2] >= self.episode_cap:
                reward = self.loss_reward
        elif a_type == "inform":
            # a is the inform
            results = self.get_inform(s)
            agent_utt = self.index_inform.get("all")
            if self.person_inmind_key in results:
                guess = self.random_state.choice(results)

                if self.person_inmind_key == guess:
                    reward = self.win_reward
                    # has informed
                    ns[0, -1] = 1
                    resp = self.index_response.get('correct')
                else:
                    slope = (self.wrong_guess_reward - self.win_reward) / len(self.corpus)
                    reward = len(results) * slope
                    resp = self.index_response.get('wrong')
            else:
                reward = self.loss_reward
                resp = self.index_response.get('wrong')
                ns[0, -1] = 1
        else:
            # computer operator, finding previous question
            prev_question = np.argwhere(s == self.holding)
            agent_utt = []
            resp = []
            if prev_question.shape[0] > 0:
                if a_type == "include":
                    ns[prev_question[0][0], prev_question[0][1]] = self.yes
                elif a_type == "exclude":
                    ns[prev_question[0][0], prev_question[0][1]] = self.no
                elif a_type == "ignore":
                    ns[prev_question[0][0], prev_question[0][1]] = self.unknown

                results = self.get_inform(s)
                if results:
                    resp = self.index_result.get(str(len(results)))
                else:
                    resp = self.index_result.get(None)

        # append the agent action and user response to the dialog hist
        nhist.extend(agent_utt)
        nhist.extend(resp)
        return reward, (ns, nhist), self.is_terminal(ns)

    def is_terminal(self, s):
        # either we already have informed or we used all the turns
        if s[0, -1] == 1 or s[0, -2] >= self.episode_cap:
            return True
        else:
            return False