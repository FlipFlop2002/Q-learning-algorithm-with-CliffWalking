import numpy as np
import gymnasium as gym
from utils_env import *
import matplotlib.pyplot as plt

environments = ["Taxi-v3", "CliffWalking-v0"]
environment = environments[1]

# wyswietlanie podczas uczenia
show_training = False

# Parametry
betha = 0.1  # lr
gamma = 0.99  # gamma
epsilon = 0.1  # parametr eksploracji
episodes = 500  # l epizodow
t_max = 100  # kroki w epizodzie


# srodowisko
env = gym.make(environment, render_mode=None if not show_training else "human")
n_states = env.observation_space.n
n_actions = env.action_space.n

# inicjalizacja macierzy Q
Q = np.zeros((n_states, n_actions))
training_error = []
episode_rewards = []

# trening
for episode in range(episodes):
    state, info = env.reset()
    done = False
    t = 0
    episode_delta_sum = 0
    episode_reward_sum = 0

    while not done and t < t_max:
        # wybieramy akcje metoda e zachlanna
        action = choose_action(state, Q, epsilon, n_actions)
        # wykonujemy wybrana akcje -> dostajemy kolejny stan i nagrode
        next_state, reward, terminated, truncated, info = env.step(action)
        # print(state, action, next_state, reward)

        # Aktualizacja Q
        best_next_action = np.argmax(Q[next_state])
        delta = reward + gamma * Q[next_state, best_next_action] - Q[state, action]
        Q[state, action] += betha * delta

        state = next_state
        t += 1

        episode_delta_sum += delta
        episode_reward_sum += reward
        done = terminated or truncated

    training_error.append(episode_delta_sum)
    episode_rewards.append(episode_reward_sum)

    if episode % 100 == 0:
        print(f"Episode {episode}")

plot_metrics(training_error, False, "Training error", "episode", "error", "tr_error.pdf")
plot_metrics(episode_rewards, False, "Rewards per episode", "episode", "reward", "rewards.pdf")

# testowanie
env = gym.make(environment, render_mode="human")
state, info = env.reset()
done = False
total_reward = 0

print(Q)
while not done:
    action = np.argmax(Q[state])
    print(action)
    next_state, reward, done, truncated, info = env.step(action)
    total_reward += reward
    state = next_state
    env.render()

print(f"total reward: {total_reward}")
env.close()
