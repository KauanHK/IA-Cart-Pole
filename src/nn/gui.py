import gymnasium
import time


def _run(env, agent) -> float:

    state, _ = env.reset()

    episode_reward = 0
    done = False
    
    while not done:
        
        action = agent.select_action(state, 0)
        next_state, reward, done, _, _ = env.step(action)
        env.render()
        state = next_state
        episode_reward += reward

        print(f'Recompensa total: {episode_reward}', end = '\r')
        time.sleep(1 / 60)

    print('Recompensa:', episode_reward)
    env.close()


def run(agent, loop_times: int = 1) -> float:

    try:
        for _ in range(loop_times):
            env = gymnasium.make("CartPole-v1", render_mode = 'human')
            _run(env, agent)
    finally:
        env.close()
