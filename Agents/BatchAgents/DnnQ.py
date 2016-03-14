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
        if dqnConfig["model"] == "seq":
            return self.get_sequential_model()
        elif dqnConfig["model"] == "graph":
            return self.get_graph_model()
        else:
            print "unknown keras model"
            exit(1)

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
        graph.add_node(Dense(dqnConfig["first_hidden"]), name="l1", input="input")
        graph.add_node(Activation('tanh'), name="l1a", input="l1")
        graph.add_node(Dropout(dqnConfig["dropout"]), name="l1dp", input="l1a")

        graph.add_node(Dense(dqnConfig["second_hidden"]), name="l2", input="l1dp")
        graph.add_node(Activation('tanh'), name="l2a", input="l2")
        graph.add_node(Dropout(dqnConfig["dropout"]), name="l2dp", input="l2a")

        graph.add_node(Dense(self.domain.actions_num), name="l3")
        graph.add_node(Activation("linear"), name="l3a")

        graph.add_output(name="output", input="l3a")

        graph.compile(optimizer='rmsprop', loss={'output':'mse'})

        print "Model created"
        return graph


