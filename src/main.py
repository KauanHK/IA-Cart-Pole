import gymnasium
from nn.hyperparameters import episodes, target_update_freq, epsilon_min, epsilon_decay
from nn.network import DQNAgent
from nn.graphic import create_graph, show_graph



def main():

    env = gymnasium.make("CartPole-v1")
    rewards_per_episode = []
    steps_done = 0
    agent = DQNAgent(env)
    epsilon = 1.0

    for ep in range(episodes):
        print(ep, end = ' - ')

        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:

            action = agent.select_action(state, epsilon)
            next_state, reward, done, *_ = env.step(action)
            
            agent.memory.append((state, action, reward, next_state, done))
            
            state = next_state
            episode_reward += reward
            
            agent.optimize_model()

            if steps_done % target_update_freq == 0:
                agent.target_net.load_state_dict(agent.policy_net.state_dict())

            steps_done += 1
        
        print(episode_reward)

        epsilon = max(epsilon_min, epsilon_decay * epsilon)        
        rewards_per_episode.append(episode_reward)
        create_graph(rewards_per_episode)
        show_graph()
    

if __name__ == "__main__":
    main()
