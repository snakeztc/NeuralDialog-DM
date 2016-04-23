root_path = '/Users/Tony/Dropbox/CMU_Grad/DialPort/NeuralDM/'
corpus_path = root_path + 'Data/top100.json'
action_path = root_path + 'Data/action_data.p'
natural_path = root_path + 'Data/bigram_usr_resp.pkl'
model_dir = root_path + 'Models/'

generalConfig = {"global_seed": 100,
                 "greedy_temp": 0.5,
                 "corpus_size": 100,
                 "use-prosample": True,
                 "q_verbal": False,
                 "save_model": False}

# Simple Yes/NO based Simulator
commandConfig = {"loss_reward": -30.0,
                 "win_reward": 30.0,
                 "step_reward": 0.0,
                 "wrong_guess_reward": -10.0,
                 "logic_error": 0.0,
                 "episode_cap": 40,
                 "discount_factor": 0.99}

# Slot filling where the agent should fill the slots
slotConfig = {"loss_reward": -30.0,
              "win_reward": 30.0,
              "step_reward": 0.0,
              "wrong_guess_reward": -5.0,
              "logic_error": -5.0,
              "episode_cap": 100,
              "max_inform": 10,
              "use_shape": True,
              'shape_upper': 2.0,
              "unknown_chance": 0.0,
              "discount_factor": 0.99}

# Hybrid  where RL and SL is combined
hybridConfig = {"loss_reward": -30.0,
                "win_reward": 30.0,
                "step_reward": 0.0,
                "wrong_guess_reward": -5.0,
                "episode_cap": 50,
                "max_inform": 10,
                "use_shape": True,
                'shape_upper': 4.0,
                "unknown_chance": 0.0,
                "discount_factor": 0.99}

########################################################
##                      MODEL                         ##
########################################################

# Oracle State
dqnConfig = {"test_interval": 5000,
             "max_sample": 120001,
             'ep_max': 1.0,
             "ep_min": 0.1,
             "ep_min_step": 70000,
             "exp_size": 120000,
             "mini_batch": 32,
             "freeze_frequency": 1000,
             "update_frequency": 4,
             "test_trial": 200,
             "doubleDQN": False,
             "l1-share": 256,
             "l1-verbal": 256,
             "l2-verbal": 128,
             "l1-computer": 256,
             "l2-computer": 128,
             "third_hidden": None,
             "dropout": 0.3}

# Word LSTM
wordDqnConfig = {"test_interval": 5000,
                 "max_sample": 100001,
                 'ep_max': 1.0,
                 "ep_min": 0.1,
                 "ep_min_step": 70000,
                 "exp_size": 100000,
                 "mini_batch": 32,
                 "freeze_frequency": 1000,
                 "update_frequency": 4,
                 "test_trial": 200,
                 "doubleDQN": False,
                 "embedding": 30,
                 "pooling": None,
                 "recurrent": "LSTM",
                 "first_hidden": 256,
                 "second_hidden": 128,
                 "dropout": 0.3}
# Turn LSTM
turnDqnConfig = {"test_interval": 5000,
                 "max_sample": 120001,
                 'ep_max': 1.0,
                 "ep_min": 0.1,
                 "ep_min_step": 70000,
                 "exp_size": 120000,
                 "mini_batch": 32,
                 "freeze_frequency": 1000,
                 "update_frequency": 4,
                 "test_trial": 200,
                 "doubleDQN": False,
                 "embedding": 30,
                 "pooling": None,
                 "recurrent": "LSTM",
                 "recurrent_size": 256,
                 "l1-verbal": 128,
                 "l2-verbal": 100,
                 "l1-computer": 128,
                 "l2-computer": 100,
                 "dropout": 0.3}

# struct Turn LSTM
structDqnConfig = {"test_interval": 5000,
                   "max_sample": 120001,
                   "ep_max": 1.0,
                   "ep_min": 0.1,
                   "ep_min_step": 70000,
                   "exp_size": 120000,
                   "mini_batch": 32,
                   "freeze_frequency": 1000,
                   "update_frequency": 4,
                   "test_trial": 200,
                   "doubleDQN": False,
                   "sys_embed": 16,
                   "usr_middle": None,
                   "usr_embed": 16,
                   "recurrent": "LSTM",
                   "recurrent_size": 256,
                   "l1-verbal": 128,
                   "l2-verbal": 100,
                   "l1-computer": 128,
                   "l2-computer": 100,
                   "dropout": 0.3}

# Hybrid LSTM
hybridDqnConfig = {"test_interval": 5000,
                   "max_sample": 120001,
                   'ep_max': 1.0,
                   "ep_min": 0.1,
                   "ep_min_step": 70000,
                   "exp_size": 120000,
                   "mini_batch": 32,
                   "freeze_frequency": 1000,
                   "update_frequency": 4,
                   "test_trial": 200,
                   "doubleDQN": False,
                   "embedding": 30,
                   "pooling": None,
                   "recurrent": "LSTM",
                   "recurrent_size": 256,
                   "l1-verbal": 128,
                   "l2-verbal": 100,
                   "l1-spl-share": 128,
                   "dropout": 0.3}