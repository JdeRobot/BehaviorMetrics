#!/usr/bin/env python
import matplotlib
import matplotlib.pyplot as plt
import gym

rewards_key = 'episode_rewards'

class LivePlot(object):
    def __init__(self, outdir, data_key=rewards_key, line_color='blue'):
        """
        Liveplot renders a graph of either episode_rewards or episode_lengths
        Args:
            outdir (outdir): Monitor output file location used to populate the graph
            data_key (Optional[str]): The key in the json to graph (episode_rewards or episode_lengths).
            line_color (Optional[dict]): Color of the plot.
        """
        self.outdir = outdir
        self.data_key = data_key
        self.line_color = line_color

        #styling options
        matplotlib.rcParams['toolbar'] = 'None'
        plt.style.use('ggplot')
        plt.xlabel("Episodes")
        plt.ylabel(data_key)
        fig = plt.gcf().canvas.set_window_title('simulation_graph')

    def plot_reward_vs_epoch(self, env):
        if self.data_key is rewards_key:
            data = gym.wrappers.Monitor.get_episode_rewards(env)
        else:
            data = gym.wrappers.Monitor.get_episode_lengths(env)

        plt.plot(data, color=self.line_color)

        # pause so matplotlib will display
        # may want to figure out matplotlib animation or use a different library in the future
        plt.pause(0.000001)

    def plot_steps_vs_epoch(self, data):
        plt.ylabel("Steps")
        plt.xlabel("Epoch")
        plt.plot(list(data.keys()), list(data.values()), color=self.line_color)
        plt.pause(0.000001)

    def full_plot(self, env, data2, mode):
        fig, (ax1, ax2) = plt.subplots(2)
        fig.suptitle('Full stats')

        data1 = gym.wrappers.Monitor.get_episode_lengths(env)

        if mode == 0:
            ax1.plot(data1, color=self.line_color)
        elif mode == 1:
            ax2.ylabel("Steps")
            ax2.xlabel("Epoch")
            ax2.plot(list(data2.keys()), list(data2.values()), color=self.line_color)
        elif mode == 2:
            ax1.plot(data1, color=self.line_color)
            ax2.plot(list(data2.keys()), list(data2.values()), color=self.line_color)
        else:
            print("No mode selected")
