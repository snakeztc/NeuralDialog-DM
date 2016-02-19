from Utils.config import *
import numpy as np
np.random.seed(global_seed)
from BatchAgent import BatchAgent
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.preprocessing import sequence
from keras.layers.core import TimeDistributedMerge
from keras.regularizers import l2, activity_l2


class LstmDnnQ(BatchAgent):

    def __init__(self, domain, representation, seed=1):
        super(LstmDnnQ, self).__init__(domain, representation, seed)

    def learn(self, experiences, max_iter=20):
        # experience is in (phi_s, a, r, phi_ns)
        exp_s = experiences[0]
        exp_ar = experiences[1]
        exp_ns = experiences[2]
        num_samples = exp_ar.shape[1]

        actions = exp_ar[:, 0]
        rewards = exp_ar[:, 1]

        phi_s = sequence.pad_sequences(exp_s)
        phi_ns = sequence.pad_sequences(exp_ns)

        # calculate the targets
        y = self.representation.Qs_phi_s(phi_s)
        nqs = self.representation.Qs_phi_s(phi_ns)
        best_nqs = np.amax(nqs, axis=1).ravel()
        targets = rewards + self.domain.discount_factor * best_nqs
        indices = [int(v+i*y.shape[1]) for i, v, in enumerate(actions)]
        # update the new y
        y.flat[indices] = targets

        if not self.representation.model:
            self.representation.model = self.init_model()

        # fit the lstm deep neural nets!!
        self.representation.model.fit(phi_s, y, batch_size=num_samples, nb_epoch=1, verbose=0)

    def init_model(self):
        print "Model input dimension " + str(self.domain.nb_words)
        print "Model output dimension " + str(self.domain.actions_num)
        hidden_size = 20

        model = Sequential()
        model.add(Embedding(self.domain.nb_words, hidden_size, mask_zero=True))
        model.add(LSTM(hidden_size, return_sequences=False))
        model.add(Dropout(0.2))

        #model.add(TimeDistributedMerge(mode='ave'))

        model.add(Dense(hidden_size, init='lecun_uniform'))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))

        model.add(Dense(self.domain.actions_num, init='lecun_uniform'))
        model.add(Activation('linear'))

        opt = RMSprop(clipvalue=1.0)
        model.compile(loss='mse', optimizer=opt)
        print "Model created"
        return model

