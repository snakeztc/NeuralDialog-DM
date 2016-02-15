import numpy as np
from BatchAgent import BatchAgent
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.optimizers import SGD


class DNNfqi(BatchAgent):

    def __init__(self, domain, representation, seed=1):
        super(DNNfqi, self).__init__(domain, representation, seed)

    def learn(self, experiences, max_iter=20):
        # experience is in (phi_s, a, r, phi_ns)
        num_samples = experiences.shape[0]
        phi_s_size = self.representation.state_features_num
        phi_s = experiences[:, 0:phi_s_size]
        actions = experiences[:, phi_s_size]
        rewards = experiences[:, phi_s_size+1]
        phi_ns = experiences[:,phi_s_size+2:]

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

        # fit the deep neural nets!!
        self.representation.model.fit(phi_s, y, batch_size=num_samples, nb_epoch=1, verbose=0)

    def init_model(self):
        print "Model input dimension " + str(self.representation.state_features_num)
        print "Model output dimension " + str(self.domain.actions_num)

        model = Sequential()
        model.add(Dense(128, init='lecun_uniform', input_shape=(self.representation.state_features_num,)))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))

        #model.add(Dense(64, init='lecun_uniform'))
        #model.add(Activation('relu'))
        #model.add(Dropout(0.2))

        model.add(Dense(self.domain.actions_num, init='lecun_uniform'))
        model.add(Activation('linear')) #linear output so we can have range of real-valued outputs

        #opt = RMSprop(clipvalue=1.0)
        opt = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=False, clipvalue=1.0)
        model.compile(loss='mse', optimizer=opt)
        print "Model created"
        return model

