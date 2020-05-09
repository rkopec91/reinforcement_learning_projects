import gym
import numpy as np
import torch as T 
import matplotlib.pyplot as plt

from Agent import Agent

def plot_curve(x, scores, eps_history, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111, label="1")
    ax2 = fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, eps_history, color="C0")
    ax.set_xlabel("Training Steps", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0,t-100):(t+1)])

    ax2.scatter(x, running_avg, color="C1")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Score', color="C1")
    ax2.yaxis.set_label_position("right")
    ax2.tick_params(axis='y', colors="C1")

    plt.savefig(filename)

if __name__ =='__main__':
    env = gym.make('CartPole-v1')
    n_games = 10000
    scores = []
    eps_history = []

    agent = Agent(lr=0.0001,
                  input_dims=env.observation_space.shape,
                  n_actions=env.action_space.n)

    for i in range(n_games):
        score = 0
        done = False
        obs = env.reset()

        while not done:
            action = agent.choose_action(obs)
            obs_, reward, done, info = env.step(action)
            score += reward
            agent.learn(obs, action, reward, obs_)
            obs = obs_
        scores.append(score)
        eps_history.append(agent.epsilon)

        if i % 100 == 0:
            avg_score = np.mean(scores[-100:])

            print_score = round(score, 2)
            print_avg = round(avg_score, 2)
            print_eps = round(agent.epsilon, 2)

            print(f'episode {i}: score = {print_score} avg score = {print_avg} \
                  epsilon = {print_eps}')

    T.save(agent.Q.state_dict(), './naive_dqn.pth.tar')

    filename = 'cartpole_naive_dqn.png'
    x = [i+1 for i in range(n_games)]

    plot_curve(x, scores, eps_history, filename)

    score = 0
    done = False
    obs = env.reset()
    agent = Agent(lr=0.0001, model='./naive_dqn.pth.tar',
                  input_dims=env.observation_space.shape,
                  n_actions=env.action_space.n)
    while not done:
        action = agent.choose_action(obs)
        obs_, reward, done, info = env.step(action)
        obs = obs_
        env.render()

    env.close()
