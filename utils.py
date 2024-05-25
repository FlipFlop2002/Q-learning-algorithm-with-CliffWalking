import numpy as np


class QLearningEnvironment():
    def __init__(self, n_actions: int, reward_matrix: np.ndarray, starting_pos: np.ndarray, final_position: np.ndarray, reset_states: list):
        self.rows, self.cols = reward_matrix.shape
        self.n_states = self.rows * self.cols
        self.n_actions = n_actions
        self.pos = starting_pos
        self.state = self.cal_state(self.pos)
        self.R = reward_matrix
        self.final_position = final_position
        self.reset_starting_pos = starting_pos
        self.Q = np.zeros((self.n_states, self.n_actions))
        self.reset_states = reset_states


    def cal_state(self, position):
        return position[0] * self.cols + position[1]


    def step(self, action):
        if action == 0:
            if self.pos[0] - 1 < 0:
                pass
            else:
                self.pos[0] -= 1
        if action == 1:
            if self.pos[1] + 1 > self.cols - 1:
                pass
            else:
                self.pos[1] += 1
        if action == 2:
            if self.pos[0] + 1 > self.rows - 1:
                pass
            else:
                self.pos[0] += 1
        if action == 3:
            if self.pos[1] - 1 < 0:
                pass
            else:
                self.pos[1] -= 1

        self.state = self.cal_state(self.pos)
        reward = self.R[self.pos[0], self.pos[1]]
        terminated = np.array_equal(self.pos, self.final_position)
        if self.state in self.reset_states:
            self.pos = self.reset_starting_pos.copy()
            self.state = self.cal_state(self.pos)
        next_state = self.state
        return next_state, reward, terminated

    def reset(self):
        self.pos = self.reset_starting_pos.copy()
        self.state = self.cal_state(self.pos)
        return self.state
