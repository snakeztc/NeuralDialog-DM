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

    str_questions = ["Q"+str(i)+"-"+qd[0] for i, qd in enumerate(question_data)]
    str_informs = {"all":'inform'}
    str_response = ['yes', 'no', 'I do not know', 'I have told you', 'correct', 'wrong']
    str_computer = ["yes_include", "yes_exclude", "no_include", "no_exclude", "ignore"]

    # find the vocab size of this world
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
    print "Find all questionable slot and related statistics"
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
    unasked = 0
    unknown = 1
    yes = 2
    no = 3
    hold_yes = 4
    hold_no = 5
    hold_unknown = 6
    state_modality = [unasked, unknown, yes, no, hold_yes, hold_no, hold_unknown]

    loss_reward = -30.0  # -50
    wrong_guess_reward = -10.0  # -10
    logic_error = -2.0  # -20 i have told you and opposite yes/no
    step_reward = 0.0
    win_reward = 30.0  # 50
    episode_cap = 40
    discount_factor = 0.99
    # each value has a question, 1 inform and 3 computer operation
    actions_num = question_count + len(str_informs) + len(str_computer)
    action_types = ["question"] * question_count + ["inform"] + str_computer
    print "Number of actions is " + str(actions_num)

    # raw state is
    # [[unasked yes no] [....] ... turn_cnt informed]
    # 0: init, 1 yes, 3 no
    # create state_space_limit
    print "Constructing the state limit for each dimension"
    statespace_limits = np.zeros((question_count, 2))
    for d in range(0, question_count):
        statespace_limits[d, 1] = len(state_modality)

    # add the extra dimension for query return size, turn count, informed not_yet/successful
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
            index_tokens = [self.vocabs.index(t) + 1 for t in q.split(" ")]
            self.index_question.append(index_tokens)

        # convert each possible inform into index list save in a dictionary
        self.index_inform = {}
        for key, inform in self.str_informs.iteritems():
            index_tokens = [self.vocabs.index(t) + 1 for t in inform.split(" ")]
            self.index_inform[key] = index_tokens

        # convert each possible response to index
        self.index_response = {}
        for resp in self.str_response:
            index_tokens = [self.vocabs.index(t) + 1 for t in resp.split(" ")]
            self.index_response[resp] = index_tokens

        # convert each computer command to index
        self.index_computer = {}
        for command in self.str_computer:
            index_tokens = [self.vocabs.index(t) + 1 for t in command.split(" ")]
            self.index_computer[command] = index_tokens

        # convert each computer response to index
        self.index_result = {}
        for idx in range(0, len(self.corpus) + 1):
            self.index_result[idx] = [self.vocabs.index(str(idx)) + 1]

        # to make the agent life easier we have a set of result have been informed
        self.wrong_keys = set()

        print "Done indexing"

    def init_user(self):
        # initialize the user here
        selected_key = self.random_state.choice(self.corpus.keys(), p=self.prob)
        selected_person = self.corpus.get(selected_key)
        self.wrong_keys = set()
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
        return list(self.search(filters) - set(self.wrong_keys))

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
        s = all_s[0]
        hist = all_s[1]
        (ns, nhist) = self.get_next_state(s, hist, aID)
        reward = self.get_reward(s, ns, aID)
        #reward += self.get_potential(s, ns)
        return reward, (ns, nhist), self.is_terminal(ns)

    def get_next_state(self, s, hist, aID):
        """
        :param s: the current hidden state
        :param hist: the dialog history
        :return: (ns, nhist)
        """
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
            true_set = self.question_data[aID][2]

            if ns[0, aID] == self.unasked:
                chosen_answer = self.person_inmind.get(slot_name)
                if chosen_answer:
                    if type(chosen_answer) != list:
                        chosen_answer = [chosen_answer]
                    # check if any of chosen answer is in the set
                    matched = False
                    for answer in chosen_answer:
                        if answer in true_set:
                            matched = True
                            ns[0, aID] = self.yes
                            resp = self.index_response.get("yes")
                            break
                    if not matched:
                        ns[0, aID] = self.hold_no
                        resp = self.index_response.get("no")
                else:
                    resp = self.index_response.get("I do not know")
                    # populate to all q_id for this slot_name
                    for q_id, qd in enumerate(self.question_data):
                        if qd[0] == slot_name:
                            ns[0, q_id] = self.unknown

        elif a_type == "inform":
            # a is the inform
            results = self.get_inform(s)
            agent_utt = self.index_inform.get("all")
            if self.person_inmind_key in results:
                guess = self.random_state.choice(results)
                if self.person_inmind_key == guess:
                    # has informed successfully
                    ns[0, -1] = 1
                    resp = self.index_response.get('correct')
                else:
                    resp = self.index_response.get('wrong')
                    self.wrong_keys.add(guess)
            else:
                resp = self.index_response.get('wrong')
        else:
            # computer operator
            agent_utt = self.index_computer.get(a_type)
            resp = []
            question_ns = ns[0, 0:self.question_count]
            if a_type == "yes_include":
                question_ns[question_ns == self.hold_yes] = self.yes
            elif a_type == "yes_exclude":
                question_ns[question_ns == self.hold_yes] = self.no
            elif a_type == "no_exclude":
                question_ns[question_ns == self.hold_no] = self.no
            elif a_type == "no_include":
                question_ns[question_ns == self.hold_no] = self.yes
            elif a_type == "ignore":
                question_ns[question_ns == self.hold_unknown] = self.unknown
            else:
                print "ERROR: unknown action type"
                exit()
            ns[0, 0:self.question_count] = question_ns

        new_results = self.get_inform(ns)
        ns[0, -3] = len(new_results) if new_results else 0
        cmp_resp = self.index_result.get(ns[0, -3])

        # append the agent action and user response to the dialog hist
        nhist.extend(agent_utt)
        nhist.extend(resp)
        nhist.extend(cmp_resp)
        return ns, nhist

    def get_reward(self, s, ns, aID):
        reward = self.step_reward
        # get action type of aID
        a_type = self.get_action_type(aID)

        # check loss condition
        if ns[0, -1] == 1: # successfully inform
            reward = self.win_reward
        elif ns[0, -1] == 0 and ns[0, -2] >= self.episode_cap:  # run out of turns or logic errors
            reward = self.loss_reward
        elif a_type == "inform" and ns[0, -1] == 0:
            reward = self.wrong_guess_reward
        elif a_type == "question" and s[0, aID] != self.unasked:
            reward = self.logic_error
        elif a_type == "yes_exclude" or a_type == "no_include":
            reward = self.logic_error
        return reward

    def get_potential(self, s, ns):
        s_potential = s[0, -3] if s[0, -3] > 0 else 200
        ns_potential = ns[0, -3] if ns[0, -3] > 0 else 200
        potential = (s_potential - ns_potential) / 10
        return potential

    def is_terminal(self, s):
        # either we already have informed or we used all the turns
        if s[0, -1] > 0 or s[0, -2] >= self.episode_cap:
            return True
        else:
            return False
