import logging
import numpy as np
import os.path
import time


class BatchAgent(object):

    logger = None
    domain = None
    representation = None
    random_state = None
    behavior_representation = None
    doubleDQN = None
    mode_path = None

    def __init__(self, domain, representation, behavior_representation, doubleDQN, seed=1):
        self.logger = logging.getLogger("hrl.BatchAgent." + self.__class__.__name__)
        self.domain = domain
        self.random_state = np.random.RandomState(seed)
        self.representation = representation
        self.behavior_representation = behavior_representation
        self.doubleDQN = doubleDQN

    def learn(self, experiences):
        """
        Sadly, we have to use the representation in terms of phi_sa r phi_ns
        :param experiences:
        :return: Nothing
        """
        # experience is in (phi_s, a, r, phi_ns)
        num_samples = experiences.mini_batch_size
        (phi_s, policy_s, actions, rewards, phi_ns, policy_ns, sample_indices) = experiences.sample_mini_batch()

        # calculate the targets
        y = self.representation.Qs_phi_s(phi_s)

        nqs = self.representation.Qs_phi_s(phi_ns)

        # append next Qs into 1 matrix
        best_nqs = np.zeros(num_samples)
        if self.doubleDQN:
            nbqs = self.behavior_representation.Qs_phi_s(phi_ns)
            max_pos = {p:np.argmax(nbqs[p], axis=1) for p in self.domain.policy_names}
            for idx, p in enumerate(policy_ns):
                best_nqs[idx] = nqs[p][idx, max_pos[p][idx]]
        else:
            for idx, p in enumerate(policy_ns):
                best_nqs[idx] = np.amax(nqs[p][idx, :])

        targets = rewards + self.domain.discount_factor * best_nqs

        # compute the TD-error
        raw_by = self.behavior_representation.Qs_phi_s(phi_s)

        # update the y and td_error
        td_error = np.zeros(num_samples)

        for policy in self.domain.policy_names:
            policy_num = self.domain.policy_action_num[policy]
            valid_mask = [i for i, v in enumerate(policy_s) if v == policy]
            indices = [int(actions[i] + i*policy_num) for i in valid_mask]
            td_error[valid_mask] = np.abs(raw_by[policy].flat[indices] - targets[valid_mask])
            y[policy].flat[indices] = targets[valid_mask]
            #raw_by[policy].flat[indices] = targets[valid_mask]

        # update the priority
        experiences.update_priority(sample_indices=sample_indices, td_error=td_error)

        if not self.behavior_representation.model:
            self.behavior_representation.model = self.init_model()
            if os.path.exists(self.mode_path):
                self.representation.model = self.init_model()

        # check if there is supervised signal
        spl_targets = experiences.get_spl_experience(sample_indices)
        if spl_targets is not None:
            for s_idx, target in zip(self.domain.spl_indexs, spl_targets):
                y[s_idx] = target
                #raw_by[s_idx] = target

        self.behavior_representation.model.train_on_batch(x=phi_s, y=y)
        #self.behavior_representation.model.train_on_batch(x=phi_s, y=raw_by)

    def update_target_model(self):
        """
        Update target model to match with behavior model
        :return:
        """
        if not self.representation.model:
            self.representation.model = self.init_model()

        # copy weights value to targets
        self.representation.model.set_weights(self.behavior_representation.model.get_weights())

    def init_model(self):
        raise NotImplementedError("Models")




