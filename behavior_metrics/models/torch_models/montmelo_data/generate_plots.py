import os
import matplotlib.pyplot as plt
import numpy as np
import json

data_list = json.load(open(os.path.join(os.getcwd(), 'data.json'),))
iters = []
pred_v = []
pred_w = []
expb_v = []
expb_w = []

for data in data_list:
    iters.append(int(data["iter"]))
    pred_v.append(data["pred_v"]*2)
    pred_w.append(data["pred_w"])
    expb_v.append(data["exp_v"])
    expb_w.append(data["exp_w"])

fig1, ax1 = plt.subplots()
ax1.plot(iters, expb_v, '-r', label='Explicit Brain')
ax1.plot(iters, pred_v, '-b', label='Torch Brain')
ax1.set_title('Comparison Linear Velocity')
leg = ax1.legend()
plt.savefig(os.path.join(os.getcwd(), 'comparison_speed.jpg'))

fig2, ax2 = plt.subplots()
ax2.plot(iters, expb_w, '-r', label='Explicit Brain')
ax2.plot(iters, pred_w, '-b', label='Torch Brain')
ax2.set_title('Comparison Angular Velocity')
leg = ax2.legend()
plt.savefig(os.path.join(os.getcwd(), 'comparison_rotation.jpg'))