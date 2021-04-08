import time
from datetime import datetime
import pickle
import gym
from brains.f1rl.utils import liveplot
import gym_gazebo
import numpy as np
from gym import logger, wrappers
from brains.f1rl.utils.ddpg import DDPGAgent
import brains.f1rl.utils.ddpg_utils.settingsDDPG as settings
import tensorflow as tf

def render():
    render_skip = 0  # Skip first X episodes.
    render_interval = 50  # Show render Every Y episodes.
    render_episodes = 10  # Show Z episodes every rendering.

    if (episode % render_interval == 0) and (episode != 0) and (episode > render_skip):
        env.render()
    elif ((episode - render_episodes) % render_interval == 0) and (episode != 0) and (episode > render_skip) and \
            (render_episodes < episode):
        env.render(close=True)

def save_model(agent):
    # Tabular RL: Tabular Q-learning basically stores the policy (Q-values) of  the agent into a matrix of shape
    # (S x A), where s are all states, a are all the possible actions. After the environment is solved, just save this
    # matrix as a csv file. I have a quick implementation of this on my GitHub under Reinforcement Learning.
    from datetime import datetime
    import pickle
    date = datetime.now()
    format = date.strftime("%Y%m%d_%H%M%S")
    file_name = "_ddpg_model_g_{}_t_{}_h_{}".format(agent.gamma, agent.tau, agent.hidden_size)
    if os.path.isdir('brains/f1rl/logs') is False:
        os.mkdir('brains/f1rl/logs')
    if os.path.isdir('brains/f1rl/logs/ddpg_models/') is False:
        os.mkdir('brains/f1rl/logs/ddpg_models/')
    file_actor = open("brains/f1rl/logs/ddpg_models/" + format + file_name + '_actor.pkl', 'wb')
    file_critic = open("brains/f1rl/logs/ddpg_models/" + format + file_name + '_critic.pkl', 'wb')
    pickle.dump(agent.get_actor_weights(), file_actor)
    pickle.dump(agent.get_critic_weights(), file_critic)

# if __name__ == '__main__':
print(settings.title)
print(settings.description)

env = gym.make('GazeboF1CameraEnvDDPG-v0')

outdir = './logs/f1_ddpg_gym_experiments/'
stats = {}  # epoch: steps

env = gym.wrappers.Monitor(env, outdir, force=True)
plotter = liveplot.LivePlot(outdir)
last_time_steps = np.ndarray(0)
stimate_step_per_lap = 4000
lap_completed = False

sess = tf.Session()

agent = DDPGAgent(sess=sess,
                state_dim=(32,32),
                action_dim=2,
                training_batch=settings.batch_size,
                epsilon_decay=settings.epsdecay,
                gamma=settings.gamma,
                tau=settings.tau)

sess.run(tf.global_variables_initializer())

agent._hard_update()

#if settings.load_model:
#    exit(1)
#
#    qlearn_file = open('logs/qlearn_models/20200826_154342_qlearn_model_e_0.988614_a_0.2_g_0.9.pkl', 'rb')
#    model = pickle.load(qlearn_file)
#    print("Number of (action, state): {}".format(len(model)))
#    qlearn.q = model
#    qlearn.alpha = settings.alpha
#    qlearn.gamma = settings.gamma
#    qlearn.epsilon = settings.epsilon
#    highest_reward = 4000
#else:
    
highest_reward = 0

total_episodes = settings.total_episodes

start_time = time.time()

print(settings.lets_go)

max_action = [12., 1.5]
min_action = [2., -1.5]

for episode in range(total_episodes):
    done = False
    lap_completed = False
    cumulated_reward = 0  # Should going forward give more reward then L/R z?
    observation = env.reset()

    state, _ = observation

    for step in range(20000):

        action = agent.predict(state)

        mod_action = action
        
        for itr in range(len(action)):
            mod_action[itr] = min_action[itr] + 0.5*(max_action[itr] - min_action[itr])*(action[itr]+1)

        # Execute the action and get feedback
        nextState, reward, done, info = env.step(mod_action)
        cumulated_reward += reward

        agent.memorize(state, action, reward, done, nextState)

        agent.train()

        if highest_reward < cumulated_reward:
            highest_reward = cumulated_reward

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
        save_model(agent)

    m, s = divmod(int(time.time() - start_time), 60)
    h, m = divmod(m, 60)
    print ("EP: " + str(episode + 1) + " - Reward: " + str(
        cumulated_reward) + " - Time: %d:%02d:%02d" % (h, m, s) + " - steps: " + str(step))

print ("\n|" + str(total_episodes) + "|" + str(agent.gamma) + "|" + str(agent.tau) + "|" + str(highest_reward) + "| PICTURE |")

l = last_time_steps.tolist()
l.sort()

print("Overall score: {:0.2f}".format(last_time_steps.mean()))
print("Best 100 score: {:0.2f}".format(reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))

env.close()
