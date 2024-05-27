import numpy as np
import gymnasium as gym
from utils import *

# e zachlanne wybieranie akcji
def choose_action(state, Q, epsilon, n_actions):
    if np.random.rand() < epsilon:
        return np.random.choice(n_actions)
    else:
        return np.argmax(Q[state])

# parametry
betha = 0.1  # lr
gamma = 0.99  # gamma
epsilon = 0.1  # parametr eksploracji przy wyborze akcji
episodes = 500  # liczba epizodow
t_max = 100  # liczba krokow w epizodzie

R_matrix = np.ones((4, 12))
R_matrix = R_matrix * -1
R_matrix[3, 1:11] = -100
print(R_matrix)
start_pos = np.array([3, 0])
final_pos = np.array([3, 11])
reset_states = [37, 38, 39, 40, 41, 42, 43, 44, 45, 46]

my_env = QLearningEnvironment(4, R_matrix, start_pos, final_pos, reset_states=reset_states)


# uczenie - Q-learning
for episode in range(episodes):
    state= my_env.reset()
    done = False
    t = 0

    while not done and t < t_max:
        # wybieramy akcje metoda e zachlanna
        action = choose_action(state, my_env.Q, epsilon, my_env.n_actions)
        # wykonujemy wybrana akcje -> dostajemy kolejny stan i nagrode
        next_state, reward, done = my_env.step(action)

        # aktualizacja Q
        best_next_action = np.argmax(my_env.Q[next_state])
        delta = reward + gamma * my_env.Q[next_state, best_next_action] - my_env.Q[state, action]
        my_env.Q[state, action] += betha * delta

        state = next_state
        t += 1

print("end of training")

# testowanie
env = gym.make('CliffWalking-v0', render_mode="human")
state, info = env.reset()
done = False
total_reward = 0

while not done:
    action = np.argmax(my_env.Q[state])
    print(action)
    next_state, reward, done, truncated, info = env.step(action)
    total_reward += reward
    state = next_state
    env.render()  # Wyświetlanie środowiska w okienku

print(f'total reward: {total_reward}')
env.close()
