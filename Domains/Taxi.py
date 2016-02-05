from Domain import Domain
import numpy as np


class Taxi(Domain):

    # domain dependent
    _map_size = 5
    # the probability that the car will drft
    _drift_prob = 0.0
    _spot_locs = [[0, 0], [0, _map_size - 1], [_map_size - 1, 0], [_map_size - 1, _map_size - 2]]
    _action_names = {"up": 0, "down": 1, "left": 2, "right": 3, "pick": 4, "drop": 5}
    _step_reward = -1
    _win_reward = 20
    _illegal_reward = -10

    # taxi domain constants
    episode_cap = 1000
    discount_factor = 0.99
    actions_num = 6 # up, down, left, right, pick, drop
    # state is (dest, src, x, y)
    statespace_limits = np.array([[0, 4], [0, 5], [0, _map_size], [0, _map_size]])
    # x 0 1 2 3 4
    # 0 R  |    G
    # 1    |
    # 2
    # 3  |   |
    # 4 Y|   |B

    def get_map_size(self):
        return self._map_size

    def get_spot_locations(self):
        return self._spot_locs

    def is_terminal(self, s):
        s_dest = s[0]
        s_src = s[1]
        if s_dest == s_src:
            return True
        else:
            return False

    def s0(self):
        while True:
            s_des = self.random_state.random_integers(0, self._map_size - 1)
            s_src =self.random_state.random_integers(0, self._map_size - 1)
            s_x = self.random_state.random_integers(0, self._map_size - 1)
            s_y = self.random_state.random_integers(0, self._map_size - 1)
            if s_src != s_des:
                return np.array([s_des, s_src, s_x, s_y])

    def can_move(self, a_xy, b_xy):
        """
        Check if the agent can move from a_xy to b_xy (in either direction) since wall
        is blocking both ways
        :param a_xy: [x, y]
        :param b_xy: [x, y]
        :return: True if can, o/w False
        """
        forbidden_set = [[[0, 1], [0, 2]],
                         [[1, 1],[1, 2]],
                         [[3, 0],[3, 1]],
                         [[4, 0],[4, 1]],
                         [[3, 2],[3, 3]],
                         [[4, 2],[4, 3]]
                         ]
        if [a_xy, b_xy] in forbidden_set or [b_xy, a_xy] in forbidden_set:
            return False
        else:
            return True

    def step(self, s, a):
        """
        We assume here s is not terminal because we return terminal if sp is terminal
        :param s:
        :param a:
        :return:The tuple (r, ns, t, p_actions) =
            (Reward [value], next observed state, isTerminal [boolean])
        """
        s_dest = s[0]
        s_src = s[1]
        s_x = s[2]
        s_y = s[3]

        # pick
        if a == self._action_names.get("pick"):
            if s_src < self._map_size-1 and [s_x, s_y] == self._spot_locs[s_src]:
                ns = np.array([s_dest, self._map_size-1, s_x, s_y])
                reward = self._step_reward
            else:
                ns = s
                reward = self._illegal_reward
            return reward, ns, self.is_terminal(ns)

        # drop action
        if a == self._action_names.get("drop"):
            if s_src == self._map_size-1 and [s_x, s_y] == self._spot_locs[s_dest]:
                ns = np.array([s_dest, s_dest, s_x, s_y])
                reward = self._win_reward
            else:
                ns = s
                reward = self._illegal_reward
            return reward, ns, self.is_terminal(ns)

        # with drift_prob the car will drift
        if self.random_state.random_sample() < self._drift_prob:
            other_actions = [x for x in range(0, 4) if x != a]
            a = self.random_state.choice(other_actions)

        # check move
        if a == self._action_names.get("up"):
            ns_x = s_x - 1
            ns_y = s_y
        elif a == self._action_names.get("down"):
            ns_x = s_x + 1
            ns_y = s_y
        elif a == self._action_names.get('left'):
            ns_x = s_x
            ns_y = s_y - 1
        elif a== self._action_names.get('right'):
            ns_x = s_x
            ns_y = s_y + 1
        else:
            raise Exception("Unknown action here: " + str(a))

        reward = self._step_reward
        # check boundary
        if ns_x < 0 or ns_x > self._map_size - 1 or ns_y < 0 or ns_y > self._map_size-1:
            ns = s
            return reward, ns, self.is_terminal(ns)

        # check wall
        if self.can_move([s_x, s_y],[ns_x, ns_y]):
            ns = np.array([s_dest, s_src, ns_x, ns_y])
            return reward, ns, self.is_terminal(ns)
        else:
            ns = s
            return reward, ns, self.is_terminal(ns)

