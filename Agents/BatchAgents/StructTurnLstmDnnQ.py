from Utils.config import generalConfig, structDqnConfig, model_dir
import numpy as np
np.random.seed(generalConfig["global_seed"])
from BatchAgent import BatchAgent
from keras.models import Graph
from keras.layers.recurrent import LSTM, GRU
from keras.layers.core import Dense, Dropout, Activation, Masking
from keras.layers.embeddings import Embedding
from keras.optimizers import RMSprop
from keras.layers.core import TimeDistributedMerge, TimeDistributedDense
from keras.layers import containers


class StructTurnLstmDnnQ(BatchAgent):
    mode_path = model_dir+"best-turn-lstm.h5"

    def init_model(self):
        return self.get_graph_model()

    def get_graph_model(self):
        print "Creating model"

        graph = Graph()
        # 3 types of input sys usr and cmp
        graph.add_input(name='usr', input_shape=(None, self.representation.state_features_num))
        graph.add_input(name='sys', input_shape=(None, self.domain.actions_num+1))
        graph.add_input(name='cmp', input_shape=(None, 1))

        # add masking
        graph.add_node(Masking(mask_value=0.0, input_shape=(None, self.representation.state_features_num)), name="usr_mask", input="usr")
        graph.add_node(Masking(mask_value=0.0, input_shape=(None, self.domain.actions_num+1)), name="sys_mask", input="sys")

        # add embedding layer for sys
        graph.add_node(TimeDistributedDense(structDqnConfig["sys_embed"], input_dim=self.domain.actions_num+1),
                       name="sys_embed", input="sys_mask")

        if structDqnConfig["usr_middle"] is None:
            graph.add_node(TimeDistributedDense(structDqnConfig["usr_embed"],
                                                input_dim=self.representation.state_features_num),
                           name="usr_embed", input="usr_mask")
        else:
            graph.add_node(TimeDistributedDense(structDqnConfig["usr_middle"], activation="linear",
                                                input_dim=self.representation.state_features_num),
                           name="usr_middle", input="usr_mask")
            graph.add_node(TimeDistributedDense(structDqnConfig["usr_embed"], activation= "tanh",
                                                input_dim=structDqnConfig["usr_middle"]),
                           name="usr_embed", input="usr_middle")

        embed_size = structDqnConfig["usr_embed"] + structDqnConfig["sys_embed"] + 1

        # for each output
        loss = {}

        # shared model
        shared_model = containers.Sequential()

        # add new mask for ["sys_embed", "usr_embed", "cmp"]
        shared_model.add(Masking(mask_value=0.0, input_shape=(None, embed_size)))

        if structDqnConfig["recurrent"] == "LSTM":
            shared_model.add(LSTM(structDqnConfig["recurrent_size"], input_dim=embed_size, return_sequences=False))
        else:
            shared_model.add(GRU(structDqnConfig["recurrent_size"], input_dim=embed_size, return_sequences=False))
        shared_model.add(Dropout(structDqnConfig["dropout"]))

        # add the shared to model to graph
        graph.add_node(shared_model, name="recurrent_layers", inputs=["sys_embed", "usr_embed", "cmp"], merge_mode='concat')

        # add the policy networks
        for p_name in self.domain.policy_names:

            graph.add_node(Dense(structDqnConfig["l1-"+p_name], activation='tanh'), name="l1-"+p_name, input="recurrent_layers")
            graph.add_node(Dropout(structDqnConfig["dropout"]), name="l1dp-"+p_name, input="l1-"+p_name)

            graph.add_node(Dense(self.domain.policy_action_num[p_name], activation='linear'), name=p_name,
                           input='l1dp-'+p_name, create_output=True)

            loss[p_name] = "mse"

        opt = RMSprop(clipvalue=1.0)
        graph.compile(optimizer=opt, loss=loss)

        print graph.summary()
        print "Model created"
        return graph






