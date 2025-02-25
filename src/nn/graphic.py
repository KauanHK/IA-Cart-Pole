import matplotlib.pyplot as plt


def create_graph(rewards_per_episode: list[float]) -> None:

    plt.plot(rewards_per_episode)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('DQN on CartPole')


def show_graph() -> None:
    plt.show()
