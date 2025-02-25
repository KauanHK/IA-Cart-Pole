import gym
from nn.hyperparameters import episodes, target_update_freq, epsilon_min, epsilon_decay
from nn.network import policy_net, target_net, memory, select_action, optimize_model


def main():

    env = gym.make("CartPole-v1")
    rewards_per_episode = []
    steps_done = 0

    for _ in range(episodes):

        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:

            action = select_action(env, state, epsilon)
            next_state, reward, done, _ = env.step(action)
            
            memory.append((state, action, reward, next_state, done))
            
            state = next_state
            episode_reward += reward
            
            optimize_model()

            if steps_done % target_update_freq == 0:
                target_net.load_state_dict(policy_net.state_dict())

            steps_done += 1

        from nn.graphic import create_graph, show_graph

        epsilon = max(epsilon_min, epsilon_decay * epsilon)        
        rewards_per_episode.append(episode_reward)
        create_graph(rewards_per_episode)
        show_graph()
    