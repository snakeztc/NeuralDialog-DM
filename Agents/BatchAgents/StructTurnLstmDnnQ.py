from Utils.config import generalConfig, turnDqnConfig, model_dir
import numpy as np
np.random.seed(generalConfig["global_seed"])
from BatchAgent import BatchAgent
from keras.models import Sequential, Graph
from keras.layers.recurrent import LSTM, GRU
from keras.layers.core import Dense, Dropout, Activation, Masking
from keras.optimizers import RMSprop
from keras.layers.core import TimeDistributedMerge, TimeDistributedDense
from keras.layers import containers


class StructTurnLstmDnnQ(BatchAgent):
    mode_path = model_dir+"best-turn-lstm.h5"

    def init_model(self):
        return self.get_graph_model()

    def get_graph_model(self):
        print "Creating model"
        embed_size = turnDqnConfig["embedding"]
        pooling_type = turnDqnConfig["pooling"]
        use_pool = pooling_type is not None

        graph = Graph()
        graph.add_input(name='input', input_shape=(None, self.representation.state_features_num))

        # for each output
        loss = {}

        # shared model
        shared_model = containers.Sequential()
        shared_model.add(Masking(mask_value=0.0, input_shape=(None, self.representation.state_features_num)))

        shared_model.add(TimeDistributedDense(embed_size, input_dim=self.representation.state_features_num))

        if turnDqnConfig["recurrent"] == "LSTM":
            shared_model.add(LSTM(turnDqnConfig["recurrent_size"], input_dim=embed_size, return_sequences=use_pool))
        else:
            shared_model.add(GRU(turnDqnConfig["recurrent_size"], input_dim=embed_size, return_sequences=use_pool))
        shared_model.add(Dropout(turnDqnConfig["dropout"]))

        if use_pool:
            shared_model.add(TimeDistributedMerge(pooling_type))

        # add the shared to model to graph
        graph.add_node(shared_model, name="recurrent_layers", input="input")

        # add the policy networks
        for p_name in self.domain.policy_names:

            graph.add_node(Dense(turnDqnConfig["l1-"+p_name], activation='tanh'), name="l1-"+p_name, input="recurrent_layers")
            graph.add_node(Dropout(turnDqnConfig["dropout"]), name="l1dp-"+p_name, input="l1-"+p_name)

            graph.add_node(Dense(self.domain.policy_action_num[p_name], activation='linear'), name=p_name,
                           input='l1dp-'+p_name, create_output=True)

            loss[p_name] = "mse"

        opt = RMSprop(clipvalue=1.0)
        graph.compile(optimizer=opt, loss=loss)

        print graph.summary()
        print "Model created"
        return graph






