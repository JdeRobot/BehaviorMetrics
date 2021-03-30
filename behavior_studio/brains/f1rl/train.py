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

current_env = settings.current_env
if current_env == "laser":
    env = gym.make('GazeboF1QlearnLaserEnv-v0')
elif current_env == "camera":
    env = gym.make('GazeboF1QlearnCameraEnv-v0')
else:
    print("NO correct env selected")


outdir = './logs/f1_qlearn_gym_experiments/'
stats = {}  # epoch: steps


env = gym.wrappers.Monitor(env, outdir, force=True)
plotter = liveplot.LivePlot(outdir)
last_time_steps = np.ndarray(0)
actions = range(env.action_space.n)
stimate_step_per_lap = 4000
lap_completed = False

qlearn = QLearn(actions=actions, alpha=0.2, gamma=0.9, epsilon=0.99)

if settings.load_model:
    exit(1)

    qlearn_file = open('logs/qlearn_models/20200826_154342_qlearn_model_e_0.988614_a_0.2_g_0.9.pkl', 'rb')
    model = pickle.load(qlearn_file)
    print("Number of (action, state): {}".format(len(model)))
    qlearn.q = model
    qlearn.alpha = settings.alpha
    qlearn.gamma = settings.gamma
    qlearn.epsilon = settings.epsilon
    highest_reward = 4000
else:
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
    observation = env.reset()

    if qlearn.epsilon > 0.05:
        qlearn.epsilon *= epsilon_discount

    state = ''.join(map(str, observation))

    for step in range(20000):

        # Pick an action based on the current state
        action = qlearn.selectAction(state)

        # Execute the action and get feedback
        observation, reward, done, info = env.step(action)
        cumulated_reward += reward

        if highest_reward < cumulated_reward:
            highest_reward = cumulated_reward

        nextState = ''.join(map(str, observation))
        qlearn.learn(state, action, reward, nextState)
        env._flush(force=True)

        if not done:
            state = nextState
        else:
            last_time_steps = np.append(last_time_steps, [int(step + 1)])
            stats[episode] = step
            break

        if stimate_step_per_lap > 4000 and not lap_completed:
            print("LAP COMPLETED!!")
            lap_completed = True

    if episode % 100 == 0 and settings.plotter_graphic:
        plotter.plot_steps_vs_epoch(stats)

    if episode % 1000 == 0 and settings.save_model:
        print("\nSaving model . . .\n")
        save_model(qlearn)

    m, s = divmod(int(time.time() - start_time), 60)
    h, m = divmod(m, 60)
    print ("EP: " + str(episode + 1) + " - epsilon: " + str(round(qlearn.epsilon, 2)) + " - Reward: " + str(
        cumulated_reward) + " - Time: %d:%02d:%02d" % (h, m, s) + " - steps: " + str(step))

print ("\n|" + str(total_episodes) + "|" + str(qlearn.alpha) + "|" + str(qlearn.gamma) + "|" + str(
    initial_epsilon) + "*" + str(epsilon_discount) + "|" + str(highest_reward) + "| PICTURE |")

l = last_time_steps.tolist()
l.sort()

print("Overall score: {:0.2f}".format(last_time_steps.mean()))
print("Best 100 score: {:0.2f}".format(reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))

env.close()
