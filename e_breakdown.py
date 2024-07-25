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
parser.add_argument('-c', type=str, choices=['bw', 'color'],
                    help='[STR] Choose black&white or colored chart')
parser.add_argument('-s', type=bool, help='[BOOL] Saves .png if True')

args = parser.parse_args()

if args.c == 'bw':
    color = ['black', 'dimgray', 'darkgray', 'silver', 'lightgray']
else:
    color = ['b', 'g', 'r', 'c', 'm', 'y'] 


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
plt.figure(figsize=(21,5.5))
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
total_weight_num = 0
wbuf_dram = True
for lyr in range(num_workload):
    total_weight_num += workload[lyr].get_num_weights()

ax1 = plt.subplot(121)
wbuf = 2*size*size/32
hw = [size, size, 8, buf, buf, wbuf]
if total_weight_num <= wbuf * 1024:
    wbuf_dram = False
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
    hw_2d.get_energy(workload[i], status, wbuf_dram)
    hw_3d_combi.get_energy_div_m(workload[i], status, wbuf_dram)
    hw_3d_combi.get_energy_div_c(workload[i], status, wbuf_dram)
    hw_3d_sep.get_energy(workload[i], status, wbuf_dram)

idx = 0
en_list = np.zeros([num_workload, 8, 5])
for r in range(num_workload):
    tmp = workload[r].sa_2d_result[idx][4:9]
    en_list[r, 0] = [tmp[0], tmp[1], sum(tmp[2]), tmp[3], tmp[4]]
    tmp = workload[r].dcim_2d_result[idx][4:9]
    en_list[r, 4] = [tmp[0], tmp[1], sum(tmp[2]), tmp[3], tmp[4]]
    tmp = workload[r].sa_3d_divM_result[idx][4:9]
    en_list[r, 1] = [tmp[0], tmp[1], sum(tmp[2]), tmp[3], tmp[4]]
    tmp = workload[r].dcim_3d_divM_result[idx][4:9]
    en_list[r, 5] = [tmp[0], tmp[1], sum(tmp[2]), tmp[3], tmp[4]]
    tmp = workload[r].sa_3d_divC_result[idx][4:9]
    en_list[r, 2] = [tmp[0], tmp[1], sum(tmp[2]), tmp[3], tmp[4]]
    tmp = workload[r].dcim_3d_divC_result[idx][4:9]
    en_list[r, 6] = [tmp[0], tmp[1], sum(tmp[2]), tmp[3], tmp[4]]
    tmp = workload[r].sa_3d_result[idx][4:9]
    en_list[r, 3] = [tmp[0], tmp[1], sum(tmp[2]), tmp[3], tmp[4]]
    tmp = workload[r].dcim_3d_result[idx][4:9]
    en_list[r, 7] = [tmp[0], tmp[1], sum(tmp[2]), tmp[3], tmp[4]]
y_list = ['PE', 'PE <-> Buffer', 'Buffer', 'Buffer <-> DRAM', 'DRAM']
x_list = ['2D_SA_BL', '3D_SA_1a', '3D_SA_1b', '3D_SA_2','2D_DCIM', '3D_DCIM_1a', '3D_DCIM_1b', '3D_DCIM_2']

area_half_1 = workload[r].dcim_3d_divM_result[idx][-2] / workload[r].sa_2d_result[idx][-2]
area_half_2 = workload[r].dcim_3d_result[idx][-2] / workload[r].sa_2d_result[idx][-2]

en_improv_wo_dram = np.sum(np.mean(en_list, axis=0)[:,0:4], axis=1)
en_improv_wo_dram /= en_improv_wo_dram[0]
en_improv_wo_dram = np.reciprocal(en_improv_wo_dram)
en_improvement = np.sum(np.mean(en_list, axis=0), axis=1)
en_improvement /= en_improvement[0]
en_improvement = np.reciprocal(en_improvement)

left = np.zeros(8)
for i in range(5):
    bar = plt.barh(x_list, np.sum(en_list[:,:,i], axis=0), left=left, label=y_list[i], color=color[i], height=0.6)
    per_list = np.mean(en_list, axis=0) 
    per_list = per_list / np.sum(per_list, axis=1, keepdims=True)
    j = 0
    if i == 4:
        for rect in bar:           
            plt.text((rect.get_width() + left[j]) + left[-1]*0.35, rect.get_y()+0.5*rect.get_height(), '{:.2f}x'.format(en_improvement[j]), 
                        ha='center', va='center', size = 15, c='black', weight='bold', rotation=0)
            j += 1
    left += np.sum(en_list[:,:,i], axis=0)

plt.xlabel('Energy (\u03BCJ)', weight='bold', size=17)
plt.xticks(size=17, weight='bold')
plt.yticks(size=15, weight='bold')
plt.title('MAC: {:.0f}K  Act.Buf: {:.0f}KB  Wt.Buf: {:.0f}KB'.format(size*size/1024, 2*buf, wbuf), weight='bold', size=17)
plt.axhline(3.5, color='r')


ax2 = plt.subplot(122)
workload = []

with open(workload_path, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    firstrow = next(reader)
    layer_num = 0
    for row in reader:
        workload.append(wk_layer(row, layer_num))
        layer_num += 1

num_workload = len(workload)   
total_weight_num = 0
wbuf_dram = True
for lyr in range(num_workload):
    total_weight_num += workload[lyr].get_num_weights()

size2 = 2 * size
buf2 = buf * 4
wbuf2 = 2*size2*size2/32
hw2 = [size2, size2, 8, buf2, buf2, wbuf2]
hw3 = [64, 32, 8, buf2, buf2, wbuf2]
if total_weight_num <= wbuf2 * 1024:
    wbuf_dram = False
hw_2d = hw_config_2d(hw2)
hw_3d_combi = hw_config_3d_combi(hw2)
hw_3d_sep = hw_config_3d_mem_on_logic(hw2)
for i in range(num_workload):
    if i == 0:
        status = 'start'
    elif i == num_workload:
        status = 'end'
    else:
        status = 'middle'
    hw_2d.get_energy(workload[i], status, wbuf_dram)
    hw_3d_combi.get_energy_div_m(workload[i], status, wbuf_dram)
    hw_3d_combi.get_energy_div_c(workload[i], status, wbuf_dram)
    hw_3d_sep.get_energy(workload[i], status, wbuf_dram)

en_list = np.zeros([num_workload, 8, 5])
for r in range(num_workload):
    tmp = workload[r].sa_2d_result[idx][4:9]
    en_list[r, 0] = [tmp[0], tmp[1], sum(tmp[2]), tmp[3], tmp[4]]
    tmp = workload[r].dcim_2d_result[idx][4:9]
    en_list[r, 4] = [tmp[0], tmp[1], sum(tmp[2]), tmp[3], tmp[4]]
    tmp = workload[r].sa_3d_divM_result[idx][4:9]
    en_list[r, 1] = [tmp[0], tmp[1], sum(tmp[2]), tmp[3], tmp[4]]
    tmp = workload[r].dcim_3d_divM_result[idx][4:9]
    en_list[r, 5] = [tmp[0], tmp[1], sum(tmp[2]), tmp[3], tmp[4]]
    tmp = workload[r].sa_3d_divC_result[idx][4:9]
    en_list[r, 2] = [tmp[0], tmp[1], sum(tmp[2]), tmp[3], tmp[4]]
    tmp = workload[r].dcim_3d_divC_result[idx][4:9]
    en_list[r, 6] = [tmp[0], tmp[1], sum(tmp[2]), tmp[3], tmp[4]]
    tmp = workload[r].sa_3d_result[idx][4:9]
    en_list[r, 3] = [tmp[0], tmp[1], sum(tmp[2]), tmp[3], tmp[4]]
    tmp = workload[r].dcim_3d_result[idx][4:9]
    en_list[r, 7] = [tmp[0], tmp[1], sum(tmp[2]), tmp[3], tmp[4]]
y_list = ['PE', 'PE ↔ Buffer', 'Buffer', 'Buffer ↔ DRAM', 'DRAM']
x_list = ['2D_SA_BL', '3D_SA_1a', '3D_SA_1b', '3D_SA_2','2D_DCIM', '3D_DCIM_1a', '3D_DCIM_1b', '3D_DCIM_2']

area_1_1 = workload[r].dcim_3d_divM_result[idx][-2] / workload[r].sa_2d_result[idx][-2]
area_1_2 = workload[r].dcim_3d_result[idx][-2] / workload[r].sa_2d_result[idx][-2]
area_1_3 = workload[r].dcim_2d_result[idx][-2] / workload[r].sa_2d_result[idx][-2]

left = np.zeros(8)
en_improv_wo_dram = np.sum(np.mean(en_list, axis=0)[:,0:4], axis=1)
en_improv_wo_dram /= en_improv_wo_dram[0]
en_improv_wo_dram = np.reciprocal(en_improv_wo_dram)
en_improvement = np.sum(np.mean(en_list, axis=0), axis=1)
en_improvement /= en_improvement[0]
en_improvement = np.reciprocal(en_improvement)

for i in range(5):
    bar = plt.barh(x_list, np.sum(en_list[:,:,i], axis=0), left=left, label=y_list[i], color=color[i], height=0.6 )
    per_list = np.mean(en_list, axis=0) / num_workload
    j = 0
    if i == 4:
        for rect in bar:           
            plt.text((rect.get_width() + left[j]) + left[-1]*0.3, rect.get_y()+0.5*rect.get_height(), '{:.2f}x'.format(en_improvement[j]), 
                        ha='center', va='center', size = 15, c='black', weight='bold', rotation=0)
            j += 1
    left += np.sum(en_list[:,:,i], axis=0)
plt.xlabel('Energy (\u03BCJ)', weight='bold', size=17)
plt.xticks(size=17, weight='bold')
plt.yticks(size=15, weight='bold')
ax1.set_xlim([0, np.max(left)*1.5])
ax2.set_xlim([0, np.max(left)*1.5])
ax1.tick_params(width=2)
ax2.tick_params(width=2)
plt.axhline(3.5, color='r')
txt="(b)"
plt.figtext(0.723, -0.05, txt, wrap=True, horizontalalignment='center', fontsize=20, weight='bold')
txt="(a)"
plt.figtext(0.3, -0.05, txt, wrap=True, horizontalalignment='center', fontsize=20, weight='bold')

plt.legend(prop={'weight':'bold', 'size': 15})
plt.title('MAC: {:.0f}K  Act.Buf: {:.0f}KB  Wt.Buf: {:.0f}KB'.format(size2*size2/1024, 2* buf2, wbuf2), weight='bold', size=17)

plt.show()

if args.s == True:
    plt.savefig('./Figures/e_breakdown.png', dpi=300, bbox_inches='tight')