from Utils.config import *
import numpy as np
np.random.seed(global_seed)
from BatchAgent import BatchAgent
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.optimizers import SGD
from keras.regularizers import l2, activity_l2


class DNNfqi(BatchAgent):

    def __init__(self, domain, representation, behavior_representation, seed=1, doubleDQN=False):
        super(DNNfqi, self).__init__(domain, representation, seed)
        self.behavior_representation = behavior_representation
        self.doubleDQN = doubleDQN


    def learn(self, experiences, max_iter=20):
        # experience is in (phi_s, a, r, phi_ns)
        num_samples = experiences.shape[0]
        phi_s_size = self.representation.state_features_num
        phi_s = experiences[:, 0:phi_s_size]
        actions = experiences[:, phi_s_size]
        rewards = experiences[:, phi_s_size+1]
        phi_ns = experiences[:,phi_s_size+2:]

        # calculate the targets
        y = self.representation.Qs_phi_s(phi_s) ## A Bug should fix in LstmDNNFQI too
        nqs = self.representation.Qs_phi_s(phi_ns)
        nbqs = self.behavior_representation.Qs_phi_s(phi_ns)

        behavior_argmax = [int(v+i*y.shape[1]) for i, v, in enumerate(np.argmax(nbqs, axis=1))]
        best_nqs = nqs.flat[behavior_argmax]
        #best_nqs = np.amax(nqs, axis=1).ravel()
        targets = rewards + self.domain.discount_factor * best_nqs
        indices = [int(v+i*y.shape[1]) for i, v, in enumerate(actions)]
        # update the new y
        y.flat[indices] = targets

        # compute the TD-error
        raw_by = self.behavior_representation.Qs_phi_s(phi_s)
        td_error = np.abs(raw_by.flat[indices] - targets)

        if self.doubleDQN:
            if not self.behavior_representation.model:
                self.behavior_representation.model = self.init_model()

            # fit the lstm deep neural nets!!
            self.behavior_representation.model.fit(phi_s, y, batch_size=num_samples, nb_epoch=1, verbose=0)
        else:
            if not self.representation.model:
                self.representation.model = self.init_model()

            # fit the lstm deep neural nets!!
            self.representation.model.fit(phi_s, y, batch_size=num_samples, nb_epoch=1, verbose=0)

        return td_error

    def update_target_model(self):
        super(DNNfqi, self).update_target_model()

        if self.doubleDQN:
            if not self.representation.model:
                self.representation.model = self.init_model()

            # copy weights value to targets
            for target_layer, behavior_layer in zip(self.representation.model.layers, self.behavior_representation.model.layers):
                target_layer.set_weights(behavior_layer.get_weights())

    def init_model(self):
        print "Model input dimension " + str(self.representation.state_features_num)
        print "Model output dimension " + str(self.domain.actions_num)

        model = Sequential()
        model.add(Dense(256, init='lecun_uniform', input_shape=(self.representation.state_features_num,)))
        model.add(Activation('tanh'))
        model.add(Dropout(0.2))

        model.add(Dense(128, init='lecun_uniform'))
        model.add(Activation('tanh'))
        model.add(Dropout(0.2))

        model.add(Dense(self.domain.actions_num, init='lecun_uniform'))
        model.add(Activation('linear')) #linear output so we can have range of real-valued outputs

        opt = RMSprop(clipvalue=1.0)
        model.compile(loss='mse', optimizer=opt)
        print model.summary()
        print "Model created"
        return model

