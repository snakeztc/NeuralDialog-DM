from Utils.config import *
import numpy as np
np.random.seed(generalConfig["global_seed"])
from BatchAgent import BatchAgent
from keras.models import Sequential
from keras.layers.recurrent import LSTM, GRU
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.preprocessing import sequence
from keras.layers.core import TimeDistributedMerge

class LstmDnnQ(BatchAgent):
    def __init__(self, domain, representation, behavior_representation, seed=1, doubleDQN=False):
        super(LstmDnnQ, self).__init__(domain, representation, seed)
        self.behavior_representation = behavior_representation
        self.doubleDQN = doubleDQN

    def learn(self, experiences, max_iter=20):
        # experience is in (phi_s, a, r, phi_ns)
        num_samples = experiences.mini_batch_size
        (phi_s, actions, rewards, phi_ns, sample_indices) = experiences.sample_mini_batch()

        # calculate the targets
        y = self.representation.Qs_phi_s(phi_s)
        nqs = self.representation.Qs_phi_s(phi_ns)
        nbqs = self.behavior_representation.Qs_phi_s(phi_ns)

        behavior_argmax = [int(v+i*y.shape[1]) for i, v, in enumerate(np.argmax(nbqs, axis=1))]
        best_nqs = nqs.flat[behavior_argmax]

        targets = rewards + self.domain.discount_factor * best_nqs
        indices = [int(v+i*y.shape[1]) for i, v, in enumerate(actions)]
        # update the new y
        y.flat[indices] = targets

        # compute the TD-error
        raw_by = self.behavior_representation.Qs_phi_s(phi_s)
        td_error = np.abs(raw_by.flat[indices] - targets)

        # update the priority
        experiences.update_priority(sample_indices=sample_indices, td_error=td_error)

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

    def update_target_model(self):
        super(LstmDnnQ, self).update_target_model()

        if self.doubleDQN:
            if not self.representation.model:
                self.representation.model = self.init_model()

            # copy weights value to targets
            for target_layer, behavior_layer in zip(self.representation.model.layers, self.behavior_representation.model.layers):
                target_layer.set_weights(behavior_layer.get_weights())

    def init_model(self):
        print "Creating model"
        embed_size = rnnDqnConfig["embedding"]
        pooling_type = rnnDqnConfig["pooling"]
        use_pool = pooling_type != None

        model = Sequential()
        model.add(Embedding(self.domain.nb_words+1, embed_size, mask_zero=(not use_pool)))

        if rnnDqnConfig["recurrent"] == "LSTM":
            model.add(LSTM(rnnDqnConfig["first_hidden"], return_sequences=use_pool))
        else:
            model.add(GRU(rnnDqnConfig["first_hidden"], return_sequences=use_pool))
        model.add(Dropout(0.2))

        if use_pool:
            model.add(TimeDistributedMerge(pooling_type))

        model.add(Dense(rnnDqnConfig["second_hidden"], init='lecun_uniform'))
        model.add(Activation('tanh'))
        model.add(Dropout(0.2))

        model.add(Dense(self.domain.actions_num, init='lecun_uniform'))
        model.add(Activation('linear'))

        opt = RMSprop(clipvalue=1.0)
        model.compile(loss='mse', optimizer=opt)
        print "Model created"
        print model.summary()
        return model

