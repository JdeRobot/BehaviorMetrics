#!/usr/bin/env python
"""
This module contains the code responsible for adding noise
It can be in the images or the motor commands

Note: Gazebo can't seem to simulate any noise other than Gaussian
Separate noise module can simulate various types of noises
"""
import numpy as np
import random

def add_noise(image, config, noise_type = "gaussian"):
    """
    Function to add noise
    """
    if noise_type == "gaussian":
        return gaussian_noise(image, config["Mean"], config["Std_Dev"])
    elif noise_type == "salt&pepper":
        return salt_and_pepper_noise(image, config["Probability"])
    else:
        return image

def salt_and_pepper_noise(image, probability = 0.2):
    """
    Salt and Pepper Noise Function
    """
    output = image.copy()

    salt = np.array([255, 255, 255], dtype='uint8')
    pepper = np.array([0, 0, 0], dtype='uint8')

    probs = np.random.random(output.shape[:2])
    output[probs < (probability / 2)] = pepper
    output[probs > 1 - (probability / 2)] = salt

    return output.astype(np.uint8)

def gaussian_noise(image, mean = 0, std_dev = 0.01):
    """
    Gaussian Noise
    """
    noise = np.random.normal(mean, std_dev, image.shape)
    noise = noise.reshape(image.shape)
    output = image + noise
    
    return output.astype(np.uint8)