import logging
import numpy as np
import os.path
import time
from Utils.config import dqnConfig


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
        #nqs = self.stack_all_qs(self.representation.Qs_phi_s(phi_ns), num_samples)

        # append next Qs into 1 matrix
        if self.doubleDQN:
            #nbqs = self.behavior_representation.Qs_phi_s(phi_ns)
            #behavior_argmax = [int(v+i*y.shape[1]) for i, v, in enumerate(np.argmax(nbqs, axis=1))]

            #nbqs = self.stack_all_qs(self.behavior_representation.Qs_phi_s(phi_ns), num_samples)
            #behavior_argmax = [int(v+i*nbqs.shape[1]) for i, v, in enumerate(np.argmax(nbqs, axis=1))]
            #best_nqs = nqs.flat[behavior_argmax]

            nbqs = self.behavior_representation.Qs_phi_s(phi_ns)
            best_nqs = np.zeros(num_samples)
            max_pos = {p:np.argmax(nbqs[p], axis=1) for p in self.domain.policy_names}
            for idx, p in enumerate(policy_ns):
                best_nqs = nqs[p][idx, max_pos[p][idx]]
        else:
            best_nqs = np.zeros(num_samples)
            for idx, p in enumerate(policy_ns):
                best_nqs[idx] = np.amax(nqs[p][idx, :])
            #best_nqs = np.amax(nqs, axis=1).ravel()

        targets = rewards + self.domain.discount_factor * best_nqs

        #indices = [int(v+i*y.shape[1]) for i, v, in enumerate(actions)]
        # update the new y
        #y.flat[indices] = targets

        # compute the TD-error
        raw_by = self.behavior_representation.Qs_phi_s(phi_s)
        #td_error = np.abs(raw_by.flat[indices] - targets)

        # update the y and td_error
        td_error = np.zeros(num_samples)

        for policy in self.domain.policy_names:
            policy_num = self.domain.policy_action_num[policy]
            valid_mask = [i for i, v in enumerate(policy_s) if v == policy]
            indices = [int(actions[i] + i*policy_num) for i in valid_mask]
            y[policy].flat[indices] = targets[valid_mask]
            td_error[valid_mask] = np.abs(raw_by[policy].flat[indices] - targets[valid_mask])

        # update the priority
        experiences.update_priority(sample_indices=sample_indices, td_error=td_error)

        if not self.behavior_representation.model:
            self.behavior_representation.model = self.init_model()
            if os.path.exists(self.mode_path):
                self.representation.model = self.init_model()

        # fit the deep neural nets!!
        #if type(phi_s) == dict:
        #    y["input"] = phi_s
        for key, val in phi_s.iteritems():
            y[key] = val
        self.behavior_representation.model.fit(y, batch_size=num_samples, nb_epoch=1, verbose=0)
        #self.behavior_representation.model.fit(phi_s, y, batch_size=num_samples, nb_epoch=1, verbose=0)

    def update_target_model(self):
        """
        Update target model to match with behavior model
        :return:
        """
        if not self.representation.model:
            self.representation.model = self.init_model()

        # copy weights value to targets
        for target_layer, behavior_layer in zip(self.representation.model.nodes.values(), self.behavior_representation.model.nodes.values()):
            target_layer.set_weights(behavior_layer.get_weights())

        #for target_layer, behavior_layer in zip(self.representation.model.layers, self.behavior_representation.model.layers):
        #    target_layer.set_weights(behavior_layer.get_weights())

    def stack_all_qs(self, Qs, num_samples):
        results = np.zeros((num_samples, self.domain.actions_num))
        base = 0
        for policy in self.domain.policy_names:
            policy_num = self.domain.policy_action_num[policy]
            results[:, base:base+policy_num] = Qs[policy]
            base += policy_num
        return results

    def init_model(self):
        raise NotImplementedError("Models")




