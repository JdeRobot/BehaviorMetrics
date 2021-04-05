import time
from datetime import datetime
import pickle
import gym
from brains.f1rl.utils import liveplot
import gym_gazebo
import numpy as np
from gym import logger, wrappers
from brains.f1rl.utils.qlearn import QLearn
import brains.f1rl.utils.settings as settings
from brains.f1rl.utils.settings import actions_set


def render():
    render_skip = 0  # Skip first X episodes.
    render_interval = 50  # Show render Every Y episodes.
    render_episodes = 10  # Show Z episodes every rendering.

    if (episode % render_interval == 0) and (episode != 0) and (episode > render_skip):
        env.render()
    elif ((episode - render_episodes) % render_interval == 0) and (episode != 0) and (episode > render_skip) and \
            (render_episodes < episode):
        env.render(close=True)

def save_model(qlearn):
    # Tabular RL: Tabular Q-learning basically stores the policy (Q-values) of  the agent into a matrix of shape
    # (S x A), where s are all states, a are all the possible actions. After the environment is solved, just save this
    # matrix as a csv file. I have a quick implementation of this on my GitHub under Reinforcement Learning.
    from datetime import datetime
    import pickle
    date = datetime.now()
    format = date.strftime("%Y%m%d_%H%M%S")
    file_name = "_qlearn_model_e_{}_a_{}_g_{}".format(qlearn.epsilon, qlearn.alpha, qlearn.gamma)
    if os.path.isdir('brains/f1rl/logs') is False:
        os.mkdir('brains/f1rl/logs')
    if os.path.isdir('brains/f1rl/logs/qlearn_models/') is False:
        os.mkdir('brains/f1rl/logs/qlearn_models/')
    file = open("brains/f1rl/logs/qlearn_models/" + format + file_name + '.pkl', 'wb')
    pickle.dump(qlearn.q, file)

# if __name__ == '__main__':
print(settings.title)
print(settings.description)

env = gym.make('GazeboF1CameraEnvDDPG-v0')

outdir = './logs/f1_qlearn_gym_experiments/'
stats = {}  # epoch: steps


env = gym.wrappers.Monitor(env, outdir, force=True)
plotter = liveplot.LivePlot(outdir)
last_time_steps = np.ndarray(0)
actions = 6
stimate_step_per_lap = 4000
lap_completed = False

qlearn = QLearn(actions=actions, alpha=0.2, gamma=0.9, epsilon=0.99)

highest_reward = 0
initial_epsilon = qlearn.epsilon

total_episodes = settings.total_episodes
epsilon_discount = settings.epsilon_discount  # Default 0.9986

start_time = time.time()

print(settings.lets_go)

for episode in range(total_episodes):
    done = False
    lap_completed = False
    cumulated_reward = 0  # Should going forward give more reward then L/R z?

    print("Env Reset Calling!")

    observation = env.reset()

    print("Env Resetted!")

    state = observation

    for step in range(20000):

        # Execute the action and get feedback
        observation, reward, done, info = env.step(env.action_space.sample())
        cumulated_reward += reward

        if highest_reward < cumulated_reward:
            highest_reward = cumulated_reward

        state = observation

        if not done:
            break

        if stimate_step_per_lap > 4000 and not lap_completed:
            print("LAP COMPLETED!!")
            lap_completed = True

    if episode % 100 == 0 and settings.plotter_graphic:
        plotter.plot_steps_vs_epoch(stats)

env.close()
