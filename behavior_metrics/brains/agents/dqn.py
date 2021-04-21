import random

import gym
import numpy as np
from gym import wrappers
from keras import backend as K
from keras import optimizers
# from keras.initializers import normal
from keras.layers import Convolution2D, Flatten, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Activation, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.models import Sequential, load_model
from keras.optimizers import SGD, Adam, RMSprop
from keras.regularizers import l2

import memory


class DeepQ:
    """
    DQN abstraction.

    As a quick reminder:
        traditional Q-learning:
            Q(s, a) += alpha * (reward(s,a) + gamma * max(Q(s') - Q(s,a))
        DQN:
            target = reward(s,a) + gamma * max(Q(s'))

    """
    def __init__(self, outputs, memorySize, discountFactor, learningRate, learnStart, img_rows, img_cols, img_channels):
        """
        Parameters:
            - outputs: output size
            - memorySize: size of the memory that will store each state
            - discountFactor: the discount factor (gamma)
            - learningRate: learning rate
            - learnStart: steps to happen before for learning. Set to 128
        """
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.img_channels = img_channels
        self.output_size = outputs
        self.memory = memory.Memory(memorySize)
        self.discountFactor = discountFactor
        self.learnStart = learnStart
        self.learningRate = learningRate


    def initNetworks(self):
        model = self.createModel()
        self.model = model


    def createModel(self):
        # Network structure must be directly changed here.
        model = Sequential()
        model.add(Convolution2D(16, (3,3), strides=(2,2), input_shape=(self.img_channels, self.img_rows, self.img_cols)))
        model.add(Activation('relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(16, (3,3), strides=(2,2)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
        model.add(Flatten())
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dense(self.output_size))
        #adam = Adam(lr=self.learningRate)
        #model.compile(loss='mse',optimizer=adam)
        model.compile(RMSprop(lr=self.learningRate), 'MSE')
        model.summary()

        return model


    def printNetwork(self):
        i = 0
        for layer in self.model.layers:
            weights = layer.get_weights()
            print("Layer {}: {}".format(i, weights))
            i += 1


    def backupNetwork(self, model, backup):
        weightMatrix = []
        for layer in model.layers:
            weights = layer.get_weights()
            weightMatrix.append(weights)
        i = 0
        for layer in backup.layers:
            weights = weightMatrix[i]
            layer.set_weights(weights)
            i += 1


    def updateTargetNetwork(self):
        self.backupNetwork(self.model, self.targetModel)

    
    def getQValues(self, state):
        # predict Q values for all the actions
        predicted = self.model.predict(state)
        return predicted[0]


    def getTargetQValues(self, state):
        predicted = self.targetModel.predict(state)
        return predicted[0]


    def getMaxQ(self, qValues):
        return np.max(qValues)


    def getMaxIndex(self, qValues):
        return np.argmax(qValues)

    
    def calculateTarget(self, qValuesNewState, reward, isFinal):
        """
        # calculate the target function
        target = reward(s,a) + gamma * max(Q(s')
        """
        if isFinal:
            return reward
        else:
#             print("Target: {}".format(reward, self.discountFactor, self.getMaxQ(qValuesNewState)))
            return reward + self.discountFactor * self.getMaxQ(qValuesNewState)


    def selectAction(self, qValues, explorationRate):
        """
        # select the action with the highest Q value
        """
        rand = random.random()
        if rand < explorationRate:
            action = np.random.randint(0, self.output_size)
        else:
            action = self.getMaxIndex(qValues)
        return action


    def selectActionByProbability(self, qValues, bias):
        qValueSum = 0
        shiftBy = 0
        for value in qValues:
            if value + shiftBy < 0:
                shiftBy = - (value + shiftBy)
        shiftBy += 1e-06

        for value in qValues:
            qValueSum += (value + shiftBy) ** bias

        probabilitySum = 0
        qValueProbabilities = []
        for value in qValues:
            probability = ((value + shiftBy) ** bias) / float(qValueSum)
            qValueProbabilities.append(probability + probabilitySum)
            probabilitySum += probability
        qValueProbabilities[len(qValueProbabilities) - 1] = 1

        rand = random.random()
        i = 0
        for value in qValueProbabilities:
            if (rand <= value):
                return i
            i += 1


    def addMemory(self, state, action, reward, newState, isFinal):
        self.memory.addMemory(state, action, reward, newState, isFinal)


    def learnOnLastState(self):
        if self.memory.getCurrentSize() >= 1:
            return self.memory.getMemory(self.memory.getCurrentSize() - 1)


    def learnOnMiniBatch(self, miniBatchSize, useTargetNetwork=True):
        # Do not learn until we've got self.learnStart samples
        if self.memory.getCurrentSize() > self.learnStart:
            # learn in batches of 128
            miniBatch = self.memory.getMiniBatch(miniBatchSize)
            X_batch = np.empty((1, self.img_channels, self.img_rows, self.img_cols), dtype = np.float64)
            Y_batch = np.empty((1,self.output_size), dtype = np.float64)
            for sample in miniBatch:
                isFinal = sample['isFinal']
                state = sample['state']
                action = sample['action']
                reward = sample['reward']
                newState = sample['newState']

                qValues = self.getQValues(state)
                if useTargetNetwork:
                    qValuesNewState = self.getTargetQValues(newState)
                else :
                    qValuesNewState = self.getQValues(newState)
                targetValue = self.calculateTarget(qValuesNewState, reward, isFinal)
                X_batch = np.append(X_batch, state.copy(), axis=0)
                Y_sample = qValues.copy()
                Y_sample[action] = targetValue
                Y_batch = np.append(Y_batch, np.array([Y_sample]), axis=0)
                if isFinal:
                    X_batch = np.append(X_batch, newState.copy(), axis=0)
                    Y_batch = np.append(Y_batch, np.array([[reward]*self.output_size]), axis=0)
            self.model.fit(X_batch, Y_batch, validation_split=0.2, batch_size = len(miniBatch), epochs=1, verbose = 0)
    
    
    def saveModel(self, path):
        self.model.save(path)


    def loadWeights(self, path):
        self.model.set_weights(load_model(path).get_weights())
