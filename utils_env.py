import numpy as np
import matplotlib.pyplot as plt

# metoda e zachlanna do wyboru akcji
def choose_action(state, Q, epsilon, n_actions):
    if np.random.rand() < epsilon:
        return np.random.choice(n_actions)
    else:
        return np.argmax(Q[state])


def plot_metrics(data: list, save: bool,  title: str, x_label:str, y_label:str, file_save_name=None):
    plt.plot(data, linewidth=0.8)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    if save:
        if file_save_name:
            plt.savefig(file_save_name)

    plt.show()
    plt.close()