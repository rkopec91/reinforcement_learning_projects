import gym
import numpy as np
from Agent import Agent
import matplotlib.pyplot as plt


if __name__ == '__main__':

    env = gym.make('LunarLander-v2')
    n_games = 1000
    load_check = False

    agent = Agent(gamma=0.99, epsilon=1.0, alpha=5e-4, input_dimension=[8], actions=4, memory_size=1000000, batch_size=64, eps_decay=5e-5, replace=100)

    if load_check:
        agent.load_models()

    scores = []
    avg_scores = []
    eps_hist = []

    for i in range(n_games):
        done = False
        score = 0
        obs = env.reset()
        while not done:
            action = agent.choose_action(obs)
            obs_, reward, done, info = env.step(action)
            score += reward
            agent.record_transition(obs, action, reward, obs_, int(done))

            agent.learn()
            obs = obs_

        scores.append(score)
        avg_score = np.mean(scores[-100:])
        avg_scores.append(avg_score)

        print('episode ', i, ' score %.1f average score %.1f epsilon %.2f' % (score, avg_score, agent.epsilon))

        eps_hist.append(agent.epsilon)

    plt.plot(range(n_games), scores, color='b')
    plt.plot(range(n_games), avg_scores, color='r')
    plt.ylabel('score')
    plt.xlabel('episode')
    plt.show()