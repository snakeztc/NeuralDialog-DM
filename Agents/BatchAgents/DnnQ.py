from Utils.config import *
import numpy as np
np.random.seed(generalConfig["global_seed"])
from BatchAgent import BatchAgent
from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
import os.path


class DnnQ(BatchAgent):
    mode_path = model_dir+"best-dqn.h5"

    def init_model(self):
        return self.get_graph_model()

    def get_sequential_model(self):
        model = Sequential()
        model.add(Dense(dqnConfig["first_hidden"], init='lecun_uniform', input_shape=(self.representation.state_features_num,)))
        model.add(Activation('tanh'))
        model.add(Dropout(dqnConfig["dropout"]))

        model.add(Dense(dqnConfig["second_hidden"], init='lecun_uniform'))
        model.add(Activation('tanh'))
        model.add(Dropout(dqnConfig["dropout"]))

        if dqnConfig["third_hidden"]:
            model.add(Dense(dqnConfig["third_hidden"], init='lecun_uniform'))
            model.add(Activation('tanh'))
            model.add(Dropout(dqnConfig["dropout"]))

        model.add(Dense(self.domain.actions_num, init='lecun_uniform'))
        model.add(Activation('linear')) #linear output so we can have range of real-valued outputs

        opt = RMSprop(clipvalue=1.0)
        model.compile(loss='mse', optimizer=opt)
        print model.summary()

        # check if we have weights
        if os.path.exists(self.mode_path):
            model.load_weights(self.mode_path)
            print "Loaded the model weights"

        print "Model created"
        return model

    def get_graph_model(self):
        print "Model input dimension " + str(self.representation.state_features_num)
        print "Model output dimension " + str(self.domain.actions_num)

        graph = Graph()
        graph.add_input(name='input', input_shape=(self.representation.state_features_num,))

        # for each output
        loss = {}

        graph.add_node(Dense(dqnConfig["l1-share"], activation="tanh"), name="l1-share", input="input")
        graph.add_node(Dropout(dqnConfig["dropout"]), name="l1dp-share", input="l1-share")

        for p_name in self.domain.policy_names:

            #graph.add_node(Dense(dqnConfig["l1-"+p_name], activation="tanh"), name="l1-"+p_name, input="input")
            #graph.add_node(Dropout(dqnConfig["dropout"]), name="l1dp-"+p_name, input="l1-"+p_name)

            graph.add_node(Dense(dqnConfig["l2-"+p_name], activation='tanh'), name="l2-"+p_name, input="l1dp-share")
            graph.add_node(Dropout(dqnConfig["dropout"]), name="l2dp-"+p_name, input="l2-"+p_name)

            graph.add_node(Dense(self.domain.policy_action_num[p_name], activation='linear'), name=p_name,
                           input='l2dp-'+p_name, create_output=True)

            loss[p_name] = "mse"

        opt = RMSprop(clipvalue=1.0)
        graph.compile(optimizer=opt, loss=loss)

        print graph.summary()
        print "Model created"
        return graph


