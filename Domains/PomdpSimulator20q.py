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
    str_computer = ["yes_include", "yes_exclude", "no_include", "no_exclude", "ignore"]

    # find the vocab size of this world
    print "Calculating the vocabulary size"
    all_utt = str_questions + str_informs.values() + str_response + str_computer
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
    print "slot names:",
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
    hold_yes = 4.0
    hold_no = 5.0
    hold_unknown = 6.0
    state_modality = [unasked, unknown, yes, no, hold_yes, hold_no, hold_unknown]

    loss_reward = -20.0
    wrong_guess_reward = -5.0
    logic_error = -2.0
    step_reward = -1.0
    win_reward = 40.0
    episode_cap = 40
    discount_factor = 0.99
    # each value has a question, 1 inform and 3 computer operation
    actions_num = question_count + len(str_informs) + len(str_computer)
    action_types = ["question"] * question_count + ["inform"] + str_computer
    print "Number of actions is " + str(actions_num)
    print "actions types are: ",
    print action_types

    # raw state is
    # [[unasked yes no] [....] ... turn_cnt informed]
    # 0: init, 1 yes, 3 no
    # create state_space_limit
    print "Constructing the state limit for each dimension"
    statespace_limits = np.zeros((question_count, 2))
    for d in range(0, question_count):
        statespace_limits[d, 1] = len(state_modality)

    # add the extra dimension for query size, turn count, informed_successful
    statespace_limits = np.vstack((statespace_limits, [0, len(corpus)]))
    statespace_limits = np.vstack((statespace_limits, [0, episode_cap]))
    statespace_limits = np.vstack((statespace_limits, [0, 2]))

    # turn count is discrete and informed is categorical
    statespace_type = [Domain.categorical] * question_count
    statespace_type.extend([Domain.discrete, Domain.discrete, Domain.categorical])

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

        # convert each computer command to index
        self.index_computer = {}
        for command in self.str_computer:
            tokens = command.split(" ")
            index_tokens = [self.vocabs.index(t) + 1 for t in tokens]
            self.index_computer[command] = index_tokens

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
        s = np.ones((1, self.statespace_size)) * self.unasked
        s[0, -3] = len(self.corpus)
        return s, [self.eos]

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

        # get action type of aID
        a_type = self.get_action_type(aID)

        # increment the counter
        ns[0, -2] = s[0, -2] + 1

        if a_type == 'question':
            agent_utt = self.index_question[aID]
            resp = self.index_response.get("I have told you")
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
                            ns[0, aID] = self.hold_yes
                            resp = self.index_response.get("yes")
                            break
                    if not matched:
                        ns[0, aID] = self.hold_no
                        resp = self.index_response.get("no")
                else:
                    for q_id, qd in enumerate(self.question_data):
                        if qd[0] == self.question_data[aID][0]:
                            ns[0, aID] = self.hold_unknown
                            resp = self.index_response.get("I do not know")
            else:
                reward = self.logic_error

        elif a_type == "inform":
            # a is the inform
            results = self.get_inform(s)
            agent_utt = self.index_inform.get("all")
            if self.person_inmind_key in results:
                guess = self.random_state.choice(results)
                ns[0, -3] = len(results)
                if self.person_inmind_key == guess:
                    reward = self.win_reward
                    # has informed
                    ns[0, -1] = 1
                    resp = self.index_response.get('correct')
                else:
                    #reward = self.wrong_guess_reward
                    slope = (self.wrong_guess_reward - self.win_reward) / len(self.corpus)
                    reward = len(results) * slope
                    resp = self.index_response.get('wrong')
            else:
                # logic error occurs
                ns[0, -3] = len(self.corpus)
                reward = self.wrong_guess_reward
                resp = self.index_response.get('wrong')
        else:
            # computer operator
            agent_utt = self.index_computer.get(a_type)
            resp = self.index_result.get(None)
            ns[0, -3] = len(self.corpus)
            question_ns = ns[0, 0:self.question_count]
            if a_type == "yes_include":
                question_ns[question_ns == self.hold_yes] = self.yes
            elif a_type == "yes_exclude":
                question_ns[question_ns == self.hold_yes] = self.no
                reward = self.logic_error
            elif a_type == "no_exclude":
                question_ns[question_ns == self.hold_no] = self.no
            elif a_type == "no_include":
                question_ns[question_ns == self.hold_no] = self.yes
                reward = self.logic_error
            elif a_type == "ignore":
                question_ns[question_ns == self.hold_unknown] = self.unknown
            else:
                print "ERROR: unknown action type"
                exit()
            ns[0, 0:self.question_count] = question_ns

            results = self.get_inform(ns)
            if results:
                resp = self.index_result.get(str(len(results)))
                ns[0, -3] = len(results)

        if a_type != "inform" and ns[0, -2] >= self.episode_cap:
            reward = self.loss_reward

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
