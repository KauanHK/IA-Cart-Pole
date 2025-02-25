import matplotlib.pyplot as plt


def create_graph() -> None:

    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('DQN on CartPole')

def update(rewards_per_episode: list[float]) -> None:
    plt.plot(rewards_per_episode)


def show_graph() -> None:
    plt.show()
