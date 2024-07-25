import numpy as np
import matplotlib.pyplot as plt
import csv
import matplotlib as mpl
import os
mpl.rcParams['lines.marker'] = '.'
plt.rcParams["font.family"] = "arial"
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import argparse

from utils import *
from hardware import *
from workload import wk_layer

parser = argparse.ArgumentParser()

parser.add_argument('-x', type=int, required=True,
                    help='[INT] x dim of compute array')
parser.add_argument('-y', type=int, required=True,
                    help='[INT] y dim of compute array')
parser.add_argument('--buf', type=int, required=True,
                    help='[INT] Total activation buffer size')
parser.add_argument('--wk', type=str, required=True,
                    help='[STR] Name of the workload')
parser.add_argument('-s', type=bool, help='[BOOL] Saves .png if True')

args = parser.parse_args()


base_path = './workloads/'
wk_list = os.listdir(base_path)
for i in range(len(wk_list)):
    wk_list[i] = wk_list[i].strip('.csv')

try:
    wk = wk_list.index(args.wk)
except ValueError:
    print('{} is not found!!!'.format(args.wk))
    exit(1)

    
size = args.x
buf = np.round(args.buf / 2)

print('Running simulation for: {}'.format(args.wk))
workload_path = base_path + wk_list[wk] + '.csv'
workload = []
    
with open(workload_path, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    firstrow = next(reader)
    layer_num = 0
    for row in reader:
        workload.append(wk_layer(row, layer_num))
        layer_num += 1

num_workload = len(workload)
hw = [size, size, 8, buf, buf, size*size*2/32]
hw_2d = hw_config_2d(hw)
hw_3d_combi = hw_config_3d_combi(hw)
hw_3d_sep = hw_config_3d_mem_on_logic(hw)
for i in range(num_workload):
    if i == 0:
        status = 'start'
    elif i == num_workload:
        status = 'end'
    else:
        status = 'middle'
    hw_2d.get_energy(workload[i], status, False)
    hw_3d_combi.get_energy_div_m(workload[i], status, False)
    hw_3d_combi.get_energy_div_c(workload[i], status, False)
    hw_3d_sep.get_energy(workload[i], status, False)

cycle_list = np.zeros([num_workload, 8, 2])
for r in range(num_workload):
    cycle_list[r, 0] = workload[r].sa_2d_result[0][1:3]
    cycle_list[r, 1] = workload[r].sa_3d_divM_result[0][1:3] 
    cycle_list[r, 2] = workload[r].sa_3d_divC_result[0][1:3]
    cycle_list[r, 3] = workload[r].sa_3d_result[0][1:3] 
    cycle_list[r, 4] = workload[r].dcim_2d_result[0][1:3]
    cycle_list[r, 5] = workload[r].dcim_3d_divM_result[0][1:3]
    cycle_list[r, 6] = workload[r].dcim_3d_divC_result[0][1:3]
    cycle_list[r, 7] = workload[r].dcim_3d_result[0][1:3]

fig = plt.figure(figsize=(10,3))
plt.subplot(121)
xlist = [i+1 for i in range(num_workload)]
bl_cycle = cycle_list[:, 0, 0] 
divM_cycle = cycle_list[:, 1, 0] 
divC_cycle = cycle_list[:, 2, 0] 
_3d_2_cycle = cycle_list[:, 3, 0] 
if wk_list[wk].strip('.csv') == 'ResNet18' :
    print('{:.2f}'.format(np.mean(bl_cycle / bl_cycle) * 100))
    print('{:.2f}'.format(np.mean(divM_cycle / bl_cycle) * 100))
    print('{:.2f}'.format(np.mean(divC_cycle / bl_cycle) * 100))
    print('{:.2f}'.format(np.mean(_3d_2_cycle / bl_cycle) * 100))
plt.xticks(np.arange(1, num_workload+1, step=2), size=17, weight='bold')
if num_workload <= 10:
    plt.xticks(np.arange(1, num_workload+1, step=1), size=17, weight='bold')
plt.plot(xlist, bl_cycle, label='2D_BL')
plt.plot(xlist, divM_cycle, label='3D_1a')
plt.plot(xlist, divC_cycle, label='3D_1b')
plt.plot(xlist, _3d_2_cycle, label='3D_2')
plt.xlabel("Layer", weight='bold', size=17)
plt.ylabel("Cycles", weight='bold', size=17)
plt.yticks([50000, 100000, 150000, 200000, 250000],['50K', '100K', '150K', '200K', '250K'],size=17, weight='bold')
plt.yticks(size=17, weight='bold')
ax = plt.gca()
ax.tick_params(width=2)    
plt.legend(ncols=2, prop={'weight':'bold', 'size': 12.5})

plt.subplot(122)

bl_util = cycle_list[:, 0, 1]
divM_util = cycle_list[:, 1, 1]
divC_util = cycle_list[:, 2, 1]
_3d_2_util = cycle_list[:, 3, 1]

# print(bl_util)
if wk_list[wk].strip('.csv') == 'Transformer' :
    print('2D:         {:.2f} %'.format(np.mean(bl_util) * 100))
    print('3D_divCout: {:.2f} %'.format(np.mean(divM_util) * 100))
    print('3D_divCin:  {:.2f} %'.format(np.mean(divC_util) * 100))
    print('3D_2:       {:.2f} %'.format(np.mean(_3d_2_util) * 100))


plt.plot(xlist, bl_util * 100, label='2D_BL')
plt.plot(xlist, divM_util * 100, label='3D_1a')
plt.plot(xlist, divC_util * 100, label='3D_1b')
plt.plot(xlist, _3d_2_util * 100, label='3D_2')
plt.ylabel("Utilization (%)", weight='bold', size=17)  # we already handled the x-label with ax1
plt.xticks(np.arange(1, num_workload+1, step=2), size=17, weight='bold')
if num_workload <= 10:
    plt.xticks(np.arange(1, num_workload+1, step=1), size=17, weight='bold')
    
plt.xlabel("Layer", weight='bold', size=17)
txt="(b)"
plt.figtext(0.525, 0.9, txt, wrap=True, horizontalalignment='center', fontsize=17, weight='bold')
txt="(a)"
plt.figtext(0.025, 0.9, txt, wrap=True, horizontalalignment='center', fontsize=17, weight='bold')
plt.ylim(top=100)

plt.yticks(size=17, weight='bold')
ax = plt.gca()
ax.tick_params(width=2)

fig.tight_layout() 
plt.show()

if args.s == True:
    plt.savefig('./Figures/cycles+util.png'.format(wk_list[wk].strip('.csv')), dpi=300, bbox_inches='tight')