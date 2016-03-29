import numpy as np
from Domain import Domain
from Utils.config import *
from Utils.domainUtil import DomainUtil
import pprint


class SlotSimulator20q (Domain):
    """
    This domain will have 31 question 1 inform and 3 slot filling actions.
    The domain will alternating between question_selection and slot_filling
    The agent is allowed to ask the same quesiton again since user may lie
    """

    # read config
    curConfig = slotConfig
    pprint.pprint(curConfig)

    # global varaible
    print "loading corpus and question data"
    corpus = DomainUtil.load_model(corpus_path)

    # a list of tuples (slot_name, question, true_value_set)
    question_data = DomainUtil.get_actions(action_path)

    # question action in natural language
    str_questions = ["Q"+str(i)+"-"+qd[0] for i, qd in enumerate(question_data)]
    str_informs = {key:'inform_'+unicode.encode(person.get('name'), 'utf8').replace(" ", "") for key, person in corpus.iteritems()}
    str_informs["none"] = "none"
    str_response = ['yes', 'no', 'I_do_not_know', 'correct', 'wrong']
    str_computer = ["c_unkown", "c_yes", "c_no"]
    str_result = [str(i) for i in range(0, len(corpus)+1)]
    question_count = len(question_data)
    inform_count = 1 # !! only 1 inform action
    computer_count = len(str_computer)

    # find the vocab size of this world (question + inform + user_response + computer_command + computer_result)
    all_utt = str_questions + str_informs.values() + str_response + str_computer + str_result
    # it will add EOS
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
    print "slot names:",
    print slot_names

    # prior distribution
    prior_dist = "uniform"
    prob = DomainUtil.get_prior_dist(prior_dist, len(corpus))
    print "Using prior distribution " + prior_dist

    # 20Q related
    unasked = 0
    hold_unknown = 1
    hold_yes = 2
    hold_no = 3
    # query related
    unknown = 0
    yes = 1
    no = 2
    # mode related
    spk_mode = 0
    slot_mode = 1
    resp_modality = [unasked, hold_unknown, hold_yes, hold_no]
    query_modality = [unknown, yes, no]

    loss_reward = curConfig["loss_reward"]
    wrong_guess_reward = curConfig["wrong_guess_reward"]
    logic_error = curConfig["logic_error"]
    step_reward = curConfig["step_reward"]
    win_reward = curConfig["win_reward"]
    episode_cap = curConfig["episode_cap"]
    discount_factor = curConfig.get("discount_factor")
    # each value has a question, 1 inform and 3 computer operation
    actions_num = question_count + inform_count + computer_count
    action_types = ["question"] * question_count + ["inform"] * inform_count + ["computer"] * computer_count
    action_to_policy = ["verbal"] * (question_count+inform_count) + ["computer"] * computer_count
    policy_action_num = {"verbal": (question_count+inform_count), "computer": computer_count}
    policy_names = DomainUtil.remove_duplicate(action_to_policy)
    prev_base = 0
    policy_bases = {}
    for p in policy_names:
        policy_bases[p] = prev_base
        prev_base += policy_action_num[p]
    print "Number of actions is " + str(actions_num)

    # raw state is
    # [[unasked yes no] [....] ... turn_cnt informed]
    # 0: init, 1 yes, 3 no
    # create state_space_limit
    statespace_limits = np.zeros((question_count*2, 2))
    for d in range(0, question_count):
        statespace_limits[d, 1] = len(resp_modality)
    for d in range(question_count, question_count*2):
        statespace_limits[d, 1] = len(query_modality)

    # prev_action, prev_answer, query return size, turn count, action_mode, inform_count  success_or_not
    statespace_limits = np.vstack((statespace_limits, [0, actions_num+1])) # previous action
    statespace_limits = np.vstack((statespace_limits, [0, len(query_modality)])) # prev_answer
    statespace_limits = np.vstack((statespace_limits, [0, len(corpus)])) # query return size
    statespace_limits = np.vstack((statespace_limits, [0, episode_cap])) # turn count
    statespace_limits = np.vstack((statespace_limits, [0, slotConfig["max_inform"]])) # inform count
    statespace_limits = np.vstack((statespace_limits, [0, 2])) # action mode
    statespace_limits = np.vstack((statespace_limits, [0, 2])) # success or not
    print "Constructing the state limit for each dimension of size " + str(statespace_limits.shape[0])


    # turn count is discrete and informed is categorical
    statespace_type = [Domain.categorical] * question_count * 2
    statespace_type.extend([Domain.categorical, # prev_action
                            Domain.categorical, # prev_answer
                            Domain.discrete, # query return size
                            Domain.discrete, # turn count
                            Domain.discrete, # inform count
                            Domain.categorical, # action mode
                            Domain.categorical]) # end
    prev_idx = statespace_limits.shape[0]-7 # prev_action is 1 based 0 leave for no_action
    pans_idx = statespace_limits.shape[0]-6 # previous answer
    comp_idx = statespace_limits.shape[0]-5
    turn_idx = statespace_limits.shape[0]-4
    icnt_idx = statespace_limits.shape[0]-3
    mode_idx = statespace_limits.shape[0]-2
    end_idx = statespace_limits.shape[0]-1

    print "Total number of questions " + str(len(question_data))
    print "Done initializing"
    print "*************************"

    # field_dim: field -> modality
    # field_dict: filed -> list(field_value)
    # lookup: field -> filed_value -> set(person)
    # state: state

    def __init__(self, seed):
        super(SlotSimulator20q, self).__init__(seed)

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

        # convert each possible user response to index
        self.index_response = {}
        for idx, resp in enumerate(self.str_response):
            index_tokens = [self.vocabs.index(t) + 1 for t in resp.split(" ")]
            self.index_response[resp] = (index_tokens, idx)

        # convert each computer command to index
        self.index_computer = []
        for command in self.str_computer:
            index_tokens = [self.vocabs.index(t) + 1 for t in command.split(" ")]
            self.index_computer.append(index_tokens)

        # convert each computer response to index
        self.index_result = {}
        for result in self.str_result:
            self.index_result[int(result)] = [self.vocabs.index(str(result)) + 1]

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

        # get init user
        (self.person_inmind_key, self.person_inmind) = self.init_user()

        # get init state
        s = np.zeros((1, self.statespace_size), dtype=int)
        s[0, self.comp_idx] = len(self.corpus)

        # get init turn
        # the action_id (1-based), user_resp_id (1-based), valid query size
        t = np.atleast_2d([0.0, 0.0, len(self.corpus)])

        return s, [self.eos], t

    def get_inform(self, s):
        filters = []
        for q_id in range(0, self.question_count):
            if s[0, self.question_count + q_id] == self.yes:
                filters.append((q_id, True))
            elif s[0, self.question_count + q_id] == self.no:
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

    def get_prev_action_type(self, prev_a):
        return self.get_action_type(prev_a-1)

    def parse_computer_command(self, aID):
        return aID - self.question_count - self.inform_count

    # Main Logic
    def step(self, all_s, aID):
        s = all_s[0]
        w_hist = all_s[1]
        t_hist = all_s[2]
        (ns, n_w_hist, n_t_hist) = self.get_next_state(s=s, w_hist=w_hist, t_hist=t_hist, aID=aID)
        reward = self.get_reward(s, ns, aID)

        if self.curConfig["use_shape"]:
            shape = self.get_reward_shape(s, ns)
        else:
            shape = 0.0

        return reward, shape, (ns, n_w_hist, n_t_hist), self.is_terminal(ns)

    def get_next_state(self, s, w_hist, t_hist, aID):
        """
        :param s: the current hidden state
        :param hist: the dialog history
        :return: (ns, nhist)
        """
        ns = np.copy(s)
        n_w_hist = list(w_hist)

        # get action type of aID
        a_type = self.get_action_type(aID)
        prev_a_type = self.get_prev_action_type(s[0, self.prev_idx])

        # increment the counter
        ns[0, self.turn_idx] = s[0, self.turn_idx] + 1
        # flip the mode
        if a_type == "question":
            ns[0, self.mode_idx] = self.slot_mode
        else:
            ns[0, self.mode_idx] = self.spk_mode

        # record action
        ns[0, self.prev_idx] = aID + 1

        if a_type == 'question':
            agent_utt = self.index_question[aID]
            resp = None
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
                            ns[0, aID] = self.hold_yes
                            ns[0, self.pans_idx] = self.yes
                            resp = self.index_response.get("yes")
                            break
                    if not matched:
                        ns[0, aID] = self.hold_no
                        ns[0, self.pans_idx] = self.no
                        resp = self.index_response.get("no")
                else:
                    resp = self.index_response.get("I_do_not_know")
                    ns[0, self.pans_idx] = self.unknown
                    # populate to all q_id for this slot_name
                    for q_id, qd in enumerate(self.question_data):
                        if qd[0] == slot_name:
                            ns[0, q_id] = self.hold_unknown
            else:
                # agent has answered this before
                if s[0, aID] == self.hold_yes:
                    resp = self.index_response.get("yes")
                elif s[0, aID] == self.hold_no:
                    resp = self.index_response.get("no")
                elif s[0, aID] == self.hold_unknown:
                    resp = self.index_response.get("I_do_not_know")
                else:
                    print "something wrong for question"
                    exit(1)

        elif a_type == "inform":
            # increment the inform count
            ns[0, self.icnt_idx] = s[0, self.icnt_idx] + 1
            # a is the inform
            results = self.get_inform(s)
            if self.person_inmind_key in results:
                guess = self.random_state.choice(results)
                agent_utt = self.index_inform.get(guess)
                if self.person_inmind_key == guess:
                    # has informed successfully
                    ns[0, self.end_idx] = 1
                    resp = self.index_response.get('correct')
                else:
                    resp = self.index_response.get('wrong')
                    self.wrong_keys.add(guess)
            else:
                agent_utt = self.index_inform.get("none")
                resp = self.index_response.get('wrong')

        elif a_type == "computer":
            if s[0, self.mode_idx] != self.slot_mode:
                print "something wrong for the action masking"
                exit(1)
            resp = ([], -1)
            # update the query
            query_val = self.parse_computer_command(aID)
            # computer operator
            agent_utt = self.index_computer[query_val]
            # update if possible
            if prev_a_type == "question":
                query_idx = self.question_count + s[0, self.prev_idx] - 1
                ns[0, query_idx] = query_val

            else:
                print "something wrong about action mask"
                exit(1)
        else:
            print "ERROR: unknown action type"
            exit(1)
            return None

        new_results = self.get_inform(ns)
        ns[0, self.comp_idx] = len(new_results) if new_results else 0
        cmp_resp = self.index_result.get(ns[0, self.comp_idx])
        if self.person_inmind_key not in new_results:
            ns[0, self.turn_idx] = self.episode_cap

        # append the agent action and user response to the dialog hist
        n_w_hist.extend(agent_utt)
        n_w_hist.extend(resp[0])
        n_w_hist.extend(cmp_resp)

        # stack turn hist
        n_t_hist = np.row_stack((t_hist, np.array([aID+1, resp[1]+1, ns[0, self.comp_idx]])))

        return ns, n_w_hist, n_t_hist

    def get_reward(self, s, ns, aID):
        reward = self.step_reward
        # get action type of aID
        a_type = self.get_action_type(aID)
        prev_a_type = self.get_prev_action_type(s[0, self.prev_idx])

        # check loss condition
        if ns[0, self.end_idx] == 1: # successfully inform
            reward = self.win_reward
        elif ns[0, self.end_idx] == 0 and self.is_terminal(ns):  # run out of turns or logic errors
            reward = self.loss_reward
        elif a_type == "inform" and ns[0, self.end_idx] == 0:
            reward = self.wrong_guess_reward
        elif a_type == "computer":
            # don't fill slot for unasked
            # if the slot filling is correct
            # don't repeat the action
            if prev_a_type != "question":
                print "something wrong about action mask"
                exit(1)
            else:
                query_idx = s[0, self.prev_idx] - 1
                query_val = self.parse_computer_command(aID)
                if ns[0, query_idx] -1 != query_val:
                    reward = self.logic_error

        return reward

    def get_reward_shape(self, s, ns):
        upper_bnd = self.statespace_limits[self.comp_idx, 1] / 2.0
        s_potential = 2.0 - s[0][self.comp_idx]/upper_bnd if s[0][self.comp_idx] > 0 else 0.0
        ns_potential = 2.0 - ns[0][self.comp_idx]/upper_bnd if ns[0][self.comp_idx] > 0 else 0.0
        # since s_potential and ns_potential should be negative here
        return self.discount_factor * ns_potential - s_potential

    def is_terminal(self, s):
        # either we already have informed or we used all the turns
        if s[0, self.end_idx] > 0 or s[0, self.turn_idx] >= self.episode_cap\
                or s[0, self.icnt_idx] >= slotConfig["max_inform"]:
            return True
        else:
            return False

    def action_prune(self, all_s):
        # check if we have any query slots asked but not filled
        s = all_s[0]
        if s[0, self.mode_idx] == self.spk_mode:
            return "verbal"
        elif s[0, self.mode_idx] == self.slot_mode:
            return "computer"
        else:
            print "state mode is wrong"
            exit(1)





