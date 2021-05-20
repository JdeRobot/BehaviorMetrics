import time
from datetime import datetime
import pickle
import torch
from torch.utils.tensorboard import SummaryWriter
import gym
from brains.f1rl.utils import liveplot
import gym_gazebo
import numpy as np
from gym import logger, wrappers
from brains.f1rl.utils.ddpg import DDPG
import brains.f1rl.utils.ddpg_utils.settingsDDPG as settings
from PIL import Image

def render():
    render_skip = 0  # Skip first X episodes.
    render_interval = 50  # Show render Every Y episodes.
    render_episodes = 10  # Show Z episodes every rendering.

    if (episode % render_interval == 0) and (episode != 0) and (episode > render_skip):
        env.render()
    elif ((episode - render_episodes) % render_interval == 0) and (episode != 0) and (episode > render_skip) and \
            (render_episodes < episode):
        env.render(close=True)

# if __name__ == '__main__':
print(settings.title)
print(settings.description)

current_env = settings.current_env
if current_env == "laser":
    env_id = "GazeboF1LaserEnvDDPG-v0"
    env = gym.make('GazeboF1LaserEnvDDPG-v0')
elif current_env == "camera":
    env_id = "GazeboF1CameraEnvDDPG-v0"
    env = gym.make('GazeboF1CameraEnvDDPG-v0')
else:
    print("NO correct env selected")

outdir = './logs/f1_ddpg_gym_experiments/'

if not os.path.exists(outdir):
    os.makedirs(outdir+'images/')

env = gym.wrappers.Monitor(env, outdir, force=True)
plotter = liveplot.LivePlot(outdir)
last_time_steps = np.ndarray(0)
stimate_step_per_lap = 4000
lap_completed = False

if settings.load_model:
    save_path = outdir+'model/'
    model_path = save_path
else:
    save_path = outdir+'model/'
    model_path = None
    
highest_reward = 0

total_episodes = settings.total_episodes

start_time = time.time()

seed = 123
save_iter = int(total_episodes/20)
render = False
writer = SummaryWriter(outdir)
env.seed(seed)

ddpg = DDPG(env_id,
            32,
            2,
            render=False,
            num_process=1,
            memory_size=1000000,
            lr_p=1e-3,
            lr_v=1e-3,
            gamma=0.99,
            polyak=0.995,
            explore_size=2000,
            step_per_iter=1000,
            batch_size=256,
            min_update_step=1000,
            update_step=50,
            action_noise=0.1,
            seed=seed)

print(settings.lets_go)

max_action = [12., 2]
min_action = [2., -2]

for episode in range(total_episodes):

    global_steps = (episode - 1) * ddpg.step_per_iter
    log = dict()
    num_steps = 0
    num_episodes = 0
    total_reward = 0
    min_episode_reward = float('inf')
    max_episode_reward = float('-inf')
    lap_completed = False
    cumulated_reward = 0  # Should going forward give more reward then L/R z?

    while num_steps < ddpg.step_per_iter:
        state = env.reset()
        # state = self.running_state(state)
        episode_reward = 0

        for t in range(1000):

            if global_steps < ddpg.explore_size:  # explore
                action = env.action_space.sample()
            else:  # action with noise
                action = ddpg.choose_action(state, ddpg.action_noise)

            mod_action = action

            for itr in range(len(action)):
                mod_action[itr] = min_action[itr] + 0.5*(max_action[itr] - min_action[itr])*(action[itr]+1)
            
            next_state, reward, done, info  = env.step(action)
            # next_state = self.running_state(next_state)
            mask = 0 if done else 1
            # ('state', 'action', 'reward', 'next_state', 'mask', 'log_prob')
            ddpg.memory.push(state, action, reward, next_state, mask, None)

            # print("Points:", info['points'])
            # print("Errors:", info['errors'])
            # observation_image = Image.fromarray(info['image'].reshape(32,32))
            # observation_image.save(outdir+'/images/obs'+str(episode)+str(t)+'.jpg')
            
            episode_reward += reward
            cumulated_reward += reward
            global_steps += 1
            num_steps += 1

            if global_steps >= ddpg.min_update_step and global_steps % ddpg.update_step == 0:
                for _ in range(ddpg.update_step):
                    batch = ddpg.memory.sample(
                        ddpg.batch_size)  # random sample batch
                    ddpg.update(batch)

            if done or num_steps >= ddpg.step_per_iter:
                if highest_reward < cumulated_reward:
                    highest_reward = cumulated_reward
                break

            state = next_state

            if num_steps  > 4000 and not lap_completed:
                print("LAP COMPLETED!!")
                lap_completed = True

        num_episodes += 1
        total_reward += episode_reward
        min_episode_reward = min(episode_reward, min_episode_reward)
        max_episode_reward = max(episode_reward, max_episode_reward)

    log['num_steps'] = num_steps
    log['num_episodes'] = num_episodes
    log['total_reward'] = total_reward
    log['avg_reward'] = total_reward / num_episodes
    log['max_episode_reward'] = max_episode_reward
    log['min_episode_reward'] = min_episode_reward

    print(f"Iter: {episode}, num steps: {log['num_steps']}, total reward: {log['total_reward']: .4f}, "
            f"min reward: {log['min_episode_reward']: .4f}, max reward: {log['max_episode_reward']: .4f}, "
            f"average reward: {log['avg_reward']: .4f}")

    # record reward information
    writer.add_scalar("total reward", log['total_reward'], episode)
    writer.add_scalar("average reward", log['avg_reward'], episode)
    writer.add_scalar("min reward", log['min_episode_reward'], episode)
    writer.add_scalar("max reward", log['max_episode_reward'], episode)
    writer.add_scalar("num steps", log['num_steps'], episode)

    if episode % save_iter == 0:
        ddpg.save(save_path)

    torch.cuda.empty_cache()
env.close()
