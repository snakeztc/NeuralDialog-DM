root_path = '/Users/Tony/Dropbox/CMU_Grad/DialPort/NeuralDM/'
corpus_path = root_path + 'Data/top100.json'
action_path = root_path + 'Data/action_data.p'
model_dir = root_path + 'Models/'

generalConfig = {"global_seed": 100,
                 "greedy_temp": 0.5,
                 "corpus_size": 1000,
                 "save_model": False}

# Simple Yes/NO based Simulator
pomdpConfig = {"loss_reward": -30.0,
               "win_reward": 30.0,
               "step_reward": 0.0,
               "wrong_guess_reward": -10.0,
               "logic_error": -10.0,
               "episode_cap": 40,
               "discount_factor": 0.99}

# Slot filling where the agent should fill the slots
slotConfig = {"loss_reward": -30.0,
              "win_reward": 30.0,
              "step_reward": 0.0,
              "wrong_guess_reward": -10.0,
              "logic_error": -10.0,
              "episode_cap": 100,
              "discount_factor": 0.99}

# The idea of this simulator is remove the need for slot filling
# just remember the database via a neural network
end2endConfig = {"loss_reward": -30.0,
                 "win_reward": 30.0,
                 "step_reward": 0.0,
                 "wrong_guess_reward": -10.0,
                 "logic_error": -5.0,
                 "episode_cap": 40,
                 "discount_factor": 0.99}
# Oracle State
dqnConfig = {"model": "graph",
             "test_interval": 5000,
             "max_sample": 100001,
             'ep_max': 1.0,
             "ep_min": 0.1,
             "ep_min_step": 70000,
             "exp_size": 100000,
             "mini_batch": 32,
             "freeze_frequency": 1000,
             "update_frequency": 4,
             "test_trial": 200,
             "doubleDQN": True,
             "first_hidden": 256,
             "second_hidden": 128,
             "third_hidden": None,
             "dropout": 0.3}
# Word LSTM
wordDqnConfig = {"test_interval": 4000,
                 "max_sample": 100001,
                 'ep_max': 1.0,
                 "ep_min": 0.1,
                 "ep_min_step": 70000,
                 "exp_size": 100000,
                 "mini_batch": 32,
                 "freeze_frequency": 1000,
                 "update_frequency": 4,
                 "test_trial": 200,
                 "doubleDQN": True,
                 "embedding": 30,
                 "pooling": None,
                 "recurrent": "LSTM",
                 "first_hidden": 256,
                 "second_hidden": 128,
                 "dropout": 0.2}
# Turn LSTM
turnDqnConfig = {"test_interval": 4000,
                 "max_sample": 100001,
                 'ep_max': 1.0,
                 "ep_min": 0.1,
                 "ep_min_step": 70000,
                 "exp_size": 100000,
                 "mini_batch": 32,
                 "freeze_frequency": 1000,
                 "update_frequency": 4,
                 "test_trial": 200,
                 "doubleDQN": True,
                 "embedding": 30,
                 "pooling": None,
                 "recurrent": "LSTM",
                 "first_hidden": 256,
                 "second_hidden": 128,
                 "dropout": 0.2}


