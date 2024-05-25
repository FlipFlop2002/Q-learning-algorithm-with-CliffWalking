import numpy as np
import gymnasium as gym

# Funkcja wybierająca akcję na podstawie polityki epsilon-greedy
def choose_action(state, Q, epsilon, n_actions):
    if np.random.rand() < epsilon:
        return np.random.choice(n_actions)
    else:
        return np.argmax(Q[state])

# Parametry
betha = 0.1  # Współczynnik uczenia
gamma = 0.99  # Współczynnik gamma
epsilon = 0.1  # Parametr eksploracji
episodes = 500  # Liczba epizodów
t_max = 100  # Maksymalna liczba kroków w epizodzie

# Flaga sterująca wyświetlaniem podczas treningu
show_training = False

# Inicjalizacja środowiska
env = gym.make('CliffWalking-v0', render_mode=None if not show_training else "human")
n_states = env.observation_space.n
n_actions = env.action_space.n
print(n_states)
print(env.observation_space)
print(n_actions)
print(env.action_space)

# Inicjalizacja macierzy Q
Q = np.zeros((n_states, n_actions))

# Pętla uczenia Q-learningu
for episode in range(episodes):
    state, info = env.reset()
    done = False
    t = 0

    while not done and t < t_max:
        # wybieramy akcje metoda e zachlanna
        action = choose_action(state, Q, epsilon, n_actions)
        # wykonujemy wybrana akcje -> dostajemy kolejny stan i nagrode
        next_state, reward, done, truncated, info = env.step(action)
        print(state, action, next_state, reward)

        # Aktualizacja Q
        best_next_action = np.argmax(Q[next_state])
        delta = reward + gamma * Q[next_state, best_next_action] - Q[state, action]
        Q[state, action] += betha * delta

        state = next_state
        t += 1

    # Opcjonalnie, można śledzić postęp
    if episode % 100 == 0:
        print(f"Episode {episode}")
        print()
        print(f"Q: {Q}")

print("Trening zakończony.")

# Testowanie wyuczonej polityki
env = gym.make('CliffWalking-v0', render_mode="human")
state, info = env.reset()
done = False
total_reward = 0

while not done:
    action = np.argmax(Q[state])
    next_state, reward, done, truncated, info = env.step(action)
    total_reward += reward
    state = next_state
    env.render()  # Wyświetlanie środowiska w okienku

print(f"Total reward: {total_reward}")
env.close()
