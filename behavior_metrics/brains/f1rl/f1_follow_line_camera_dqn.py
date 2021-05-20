#!/usr/bin/env python

'''
Based on:
=======
https://github.com/vmayoral/basic_reinforcement_learning
https://gist.github.com/wingedsheep/4199594b02138dd427c22a540d6d6b8d
'''
import json
import os
import time
from distutils.dir_util import copy_tree

import gym
import gym_gazebo
import matplotlib
from gym import logger, wrappers
from keras import backend as K

from brains.f1rl.utils.settings import my_board
from brains.f1rl.utils.dqn import DeepQ


# To equal the inputs, we set the channels first and the image next.
K.set_image_data_format('channels_first')


def detect_monitor_files(training_dir):
    return [os.path.join(training_dir, f) for f in os.listdir(training_dir) if f.startswith('openaigym')]


def clear_monitor_files(training_dir):
    files = detect_monitor_files(training_dir)
    if len(files) == 0:
        return
    for file in files:
        os.unlink(file)


def plot_durations(episode_durations, step):
    import matplotlib.pyplot as plt
    import numpy as np
    
    plt.figure(1)
    plt.clf()
    
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(episode_durations)
    
    # Take 100 episode averages and plot them too
    if step % 10 == 0:
        mean_episode = np.mean(episode_durations)
        plt.plot(mean_episode)

    plt.pause(0.001)  # pause a bit so that plots are updated



episode_durations = []


####################################################################################################################
# MAIN PROGRAM
####################################################################################################################
#if __name__ == '__main__':

#REMEMBER!: turtlebot_cnn_setup.bash must be executed.
env = gym.make('GazeboF1CameraEnvDQN-v0')
outdir = './logs/f1_gym_experiments/'

current_file_path = os.path.abspath(os.path.dirname(__file__))

print("=====================\nENV CREATED\n=====================")

continue_execution = False
# Fill this if continue_execution=True
weights_path = os.path.join(current_file_path, 'logs/f1_dqn_ep27000.h5')
monitor_path = os.path.join(current_file_path, 'logs/f1_dqn_ep27000')
params_json  = os.path.join(current_file_path, 'logs/f1_dqn_ep27000.json')

img_rows, img_cols, img_channels = env.img_rows, env.img_cols, env.img_channels

epochs = 10000000
steps = 1000000

if not continue_execution:
    minibatch_size = 32
    learningRate = 1e-3  # 1e6
    discountFactor = 0.95
    network_outputs = 5
    memorySize = 100000
    learnStart = 10000  # timesteps to observe before training (default: 10.000)
    EXPLORE = memorySize  # frames over which to anneal epsilon
    INITIAL_EPSILON = 1  # starting value of epsilon
    FINAL_EPSILON = 0.01  # final value of epsilon
    explorationRate = INITIAL_EPSILON
    current_epoch = 0
    stepCounter = 0
    loadsim_seconds = 0

    deepQ = DeepQ(network_outputs, memorySize, discountFactor, learningRate, learnStart, img_rows, img_cols, img_channels)
    deepQ.initNetworks()
    env = gym.wrappers.Monitor(env, outdir, force=True)
else:
    # Load weights, monitor info and parameter info.
    with open(params_json) as outfile:
        d = json.load(outfile)
        explorationRate = d.get('explorationRate')
        minibatch_size = d.get('minibatch_size')
        learnStart = d.get('learnStart')
        learningRate = d.get('learningRate')
        discountFactor = d.get('discountFactor')
        memorySize = d.get('memorySize')
        network_outputs = d.get('network_outputs')
        current_epoch = d.get('current_epoch')
        stepCounter = d.get('stepCounter')
        EXPLORE = d.get('EXPLORE')
        INITIAL_EPSILON = d.get('INITIAL_EPSILON')
        FINAL_EPSILON = d.get('FINAL_EPSILON')
        loadsim_seconds = d.get('loadsim_seconds')

    deepQ = DeepQ(network_outputs, memorySize, discountFactor, learningRate, learnStart, img_rows, img_cols, img_channels)
    deepQ.initNetworks()
    deepQ.loadWeights(weights_path)

    clear_monitor_files(outdir)
    copy_tree(monitor_path,outdir)
    env = gym.wrappers.Monitor(env, outdir, resume=True)


last100Rewards = [0] * 100
last100RewardsIndex = 0
last100Filled = False

start_time = time.time()

# Start iterating from 'current epoch'.
for epoch in range(current_epoch+1, epochs+1, 1):

    observation, pos = env.reset()

    cumulated_reward = 0

    # Number of timesteps
    for step in range(steps):

        # make the model.predict

        observation = observation.reshape(1, 32, 32, 1)
        qValues = deepQ.getQValues(observation)
        action = deepQ.selectAction(qValues, explorationRate)
        newObservation, reward, done, _ = env.step(action)
        deepQ.addMemory(observation, action, reward, newObservation, done)

        # print("Step: {}".format(t))
        # print("Action: {}".format(action))
        # print("Reward: {}".format(reward))

        observation = newObservation

        # We reduced the epsilon gradually
        if explorationRate > FINAL_EPSILON and stepCounter > learnStart:
            explorationRate -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        if stepCounter == learnStart:
            print("Starting learning")

        if stepCounter >= learnStart:
            deepQ.learnOnMiniBatch(minibatch_size, False)

        if (step == steps-1):
            print("reached the end")
            done = True

        env._flush(force=True)
        cumulated_reward += reward

        if done:
            episode_durations.append(step)
            if my_board:
                plot_durations(episode_durations, step)

            last100Rewards[last100RewardsIndex] = cumulated_reward
            last100RewardsIndex += 1
            if last100RewardsIndex >= 100:
                last100Filled = True
                last100RewardsIndex = 0
            m, s = divmod(int(time.time() - start_time + loadsim_seconds), 60)
            h, m = divmod(m, 60)
            if not last100Filled:
                print("EP: {} - Steps: {} - Pos: {} - CReward: {} - Eps: {} - Time: {}:{}:{} ".format(epoch, step+1, pos, round(cumulated_reward, 2), round(explorationRate, 2), h, m, s))
            else:
                print("EP: {} - Steps: {} - Pos: {} - last100 C_Rewards: {} - CReward: {} - Eps={} - Time: {}:{}:{}".format(epoch, step+1, pos, sum(last100Rewards)/len(last100Rewards), round(cumulated_reward, 2), round(explorationRate, 2), h, m, s))

                # SAVE SIMULATION DATA
                if (epoch)%1000==0:
                    # Save model weights and monitoring data every 100 epochs.
                    deepQ.saveModel('./logs/f1_dqn_ep'+str(epoch)+'.h5')
                    env._flush()
                    copy_tree(outdir,'./logs/f1_dqn_ep'+str(epoch))
                    # Save simulation parameters.
                    parameter_keys = ['explorationRate','minibatch_size','learnStart','learningRate','discountFactor','memorySize','network_outputs','current_epoch','stepCounter','EXPLORE','INITIAL_EPSILON','FINAL_EPSILON','loadsim_seconds']
                    parameter_values = [explorationRate, minibatch_size, learnStart, learningRate, discountFactor, memorySize, network_outputs, epoch, stepCounter, EXPLORE, INITIAL_EPSILON, FINAL_EPSILON,s]
                    parameter_dictionary = dict(zip(parameter_keys, parameter_values))
                    with open('./logs/f1_dqn_ep'+str(epoch)+'.json', 'w') as outfile:
                        json.dump(parameter_dictionary, outfile)

            break

        stepCounter += 1
        # if stepCounter % 250 == 0:
        #     print("Frames = " + str(stepCounter))

env.close()
