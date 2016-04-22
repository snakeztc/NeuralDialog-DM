from Utils.config import *
import numpy as np
np.random.seed(generalConfig["global_seed"])
from BatchAgent import BatchAgent
from keras.models import Model
from keras.layers import Dense, Dropout, Input
from keras.optimizers import RMSprop


class DnnQ(BatchAgent):
    mode_path = model_dir+"best-dqn.h5"

    def init_model(self):
        return self.get_graph_model()

    def get_graph_model(self):

        # for each output
        loss = {}
        outputs = []


        model_input = Input(shape=(self.representation.state_features_num,), name='input')
        l1 = Dense(dqnConfig["l1-share"], activation="tanh", name="l1-share")(model_input)
        d1l = Dropout(dqnConfig["dropout"])(l1)

        # add the policy networks
        for p_idx, p_name in zip(self.domain.policy_names, self.domain.policy_str_name):
            l2 = Dense(dqnConfig["l2-"+p_name], activation='tanh', name="l2-"+p_name)(d1l)
            dl2 = Dropout(dqnConfig["dropout"], name="l2dp-"+p_name)(l2)

            o = Dense(self.domain.policy_action_num[p_idx], activation='linear', name=p_name)(dl2)
            loss[p_name] = "mse"
            outputs.append(o)

        opt = RMSprop(clipvalue=1.0)
        model = Model(input=model_input, output=outputs)
        model.compile(optimizer=opt, loss=loss)

        print model.summary()
        print "Model created"
        return model


