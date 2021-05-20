import time
import datetime
import pickle
import gym
import gym_gazebo
import numpy as np
from gym import logger, wrappers

import brains.f1rl.utils.settings as settings
from brains.f1rl.utils.qlearn import QLearn
from brains.f1rl.utils.settings import actions_set
from brains.f1rl.utils import liveplot

class Brain:
    """Specific brain for the f1 robot with q learning."""
    
    def __init__(self, sensors, actuators, handler=None):
        """Constructor of the class.

        Arguments:
            sensors {robot.sensors.Sensors} -- Sensors instance of the robot
            actuators {robot.actuators.Actuators} -- Actuators instance of the robot

        Keyword Arguments:
            handler {brains.brain_handler.Brains} -- Handler of the current brain. Communication with the controller
            (default: {None})
        """
        self.camera = sensors.get_camera('camera_0')
        # self.laser = sensors.get_laser('laser_0')
        self.pose = sensors.get_pose3d('pose3d_0')
        self.motors = actuators.get_motor('motors_0')
        self.handler = handler
        
    def update_frame(self, frame_id, data):
        """Update the information to be shown in one of the GUI's frames.

        Arguments:
            frame_id {str} -- Id of the frame that will represent the data
            data {*} -- Data to be shown in the frame. Depending on the type of frame (rgbimage, laser, pose3d, etc)
        """
        self.handler.update_frame(frame_id, data)

    def update_pose(self, pose_data):
        """Update the pose 3D information obtained from the robot.

        Arguments:
            data {*} -- Data to be updated, will be retrieved later by the UI.
        """
        self.handler.update_pose3d(pose_data)    

    def render():
        render_skip = 0  # Skip first X episodes.
        render_interval = 50  # Show render Every Y episodes.
        render_episodes = 10  # Show Z episodes every rendering.

        if (episode % render_interval == 0) and (episode != 0) and (episode > render_skip):
            env.render()
        elif ((episode - render_episodes) % render_interval == 0) and (episode != 0) and (episode > render_skip) and \
                (render_episodes < episode):
            env.render(close=True)
            
#     def execute(self):
#         """Main loop of the brain. This will be called iteratively each TIME_CYCLE (see pilot.py)"""

#         self.update_pose(self.pose.getPose3d())
#         v = 0
#         w = 0.8
#         self.motors.sendV(v)
#         self.motors.sendW(w)
#         image = self.camera.getImage().data
#         self.update_frame('frame_0', image)
#         # laser_data = self.laser.getLaserData()
#         # self.update_frame('frame_0', laser_data)
        


    def save_model():
        # Tabular RL: Tabular Q-learning basically stores the policy (Q-values) of  the agent into a matrix of shape
        # (S x A), where s are all states, a are all the possible actions. After the environment is solved, just save this
        # matrix as a csv file. I have a quick implementation of this on my GitHub under Reinforcement Learning.
        date = datetime.datetime.now()
        format = date.strftime("%Y%m%d_%H%M%S")
        file_name = "_qlearn_model_e_{}_a_{}_g_{}".format(qlearn.epsilon, qlearn.alpha, qlearn.gamma)
        file = open("logs/qlearn_models/" + format + file_name + '.pkl', 'wb')
        pickle.dump(qlearn.q, file)

        print(qlearn.q)

    def execute(self):
        current_env = "laser"
        if current_env == "laser":
            env = gym.make('GazeboF1QlearnLaserEnv-v0')
        elif current_env == "camera":
            env = gym.make('GazeboF1QlearnCameraEnv-v0')
        else:
            print("NO correct env selected")

        outdir = './logs/f1_qlearn_gym_experiments/'

        env = gym.wrappers.Monitor(env, outdir, force=True)
        plotter = liveplot.LivePlot(outdir)

        last_time_steps = np.ndarray(0)

        actions = range(len(env.action_space))
        qlearn = QLearn(actions=actions, alpha=0.2, gamma=0.9, epsilon=0.99)

        if settings.load_model:
            exit(1)
            qlearn_file = open('logs/qlearn_models/20200628_LASER_5ACTS_155827qlearn_model_e_0.243550191511_a_0.2_g_0.9.pkl', 'rb')
            model = pickle.load(qlearn_file)
            print("Number of (action, state): {}".format(len(model)))
            qlearn.q = model
            qlearn.alpha = 0.2
            qlearn.gamma = 0.9
            qlearn.epsilon = 0.6
            highest_reward = 4000
        else:
            highest_reward = 0
            initial_epsilon = qlearn.epsilon

        total_episodes = 20000
        epsilon_discount = 0.9986  # Default 0.9986

        start_time = time.time()
        for episode in range(total_episodes):

            done = False
            cumulated_reward = 0  # Should going forward give more reward then L/R z?

            observation = env.reset()

            if qlearn.epsilon > 0.05:
                qlearn.epsilon *= epsilon_discount

            # render()  # defined above, not env.render()

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
    #             print(nextState)

                qlearn.learn(state, action, reward, nextState)

                env._flush(force=True)

                if not done:
                    state = nextState
                else:
                    last_time_steps = np.append(last_time_steps, [int(step + 1)])
                    break

                if step > 3000:
                    print("\n\nLAP COMPLETED!!\n\n")

                # print("Obser: {} - Rew: {}".format(observation, reward))

            if episode % 100 == 0:
                plotter.plot(env)
                if settings.save_model:
                    print("\nSaving model . . .\n")
                    save_model()

            m, s = divmod(int(time.time() - start_time), 60)
            h, m = divmod(m, 60)
            print ("EP: " + str(episode + 1) + " - epsilon: " + str(round(qlearn.epsilon, 2)) + "] - Reward: " + str(
                cumulated_reward) + " - Time: %d:%02d:%02d" % (h, m, s) + " - steps: " + str(step))

        print ("\n|" + str(total_episodes) + "|" + str(qlearn.alpha) + "|" + str(qlearn.gamma) + "|" + str(
            initial_epsilon) + "*" + str(epsilon_discount) + "|" + str(highest_reward) + "| PICTURE |")

        l = last_time_steps.tolist()
        l.sort()

        # print("Parameters: a="+str)
        print("Overall score: {:0.2f}".format(last_time_steps.mean()))
        print("Best 100 score: {:0.2f}".format(reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))

        env.close()
        
# if __name__ == '__main__':
print("amskdjaskdfjakj retesting damn it work  pls!")
self.execute()