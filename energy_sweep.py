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

parser.add_argument('-c', type=str, choices=['bw', 'color'],
                    help='[STR] Choose black&white or colored chart')
parser.add_argument('-s', type=bool, help='[BOOL] Saves .png if True')


args = parser.parse_args()

if args.c == 'bw':
    color = ['black', 'dimgray', 'silver']
else:
    color = ['b', 'g', 'y'] 

blue_patch = mpatches.Patch(color='b', label='PE')
green_patch = mpatches.Patch(color='g', label='Buffer')
yellow_patch = mpatches.Patch(color='y', label='DRAM')
base_path = './workloads/'
wk_list = os.listdir(base_path)

tot_en = np.zeros(3)
z = 1
plt.figure(figsize=(25, 12))
for wk in range(len(wk_list)):
    wk = (wk + 8) % 8
    workload_path = base_path + wk_list[wk]
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
    
    plt.subplot(2, 4, z)
    result_arr = np.zeros([5,8,3])
    size = 16
    hw_list = []
    for i in range(5):
        buf = 64
        for j in range(5):
            wbuf = 2*size*size/32
            hw = [size, size, 8, buf, buf, wbuf]
            if total_weight_num <= wbuf * 1024:
                wbuf_dram = False
            hw_2d = hw_config_2d(hw)
            hw_3d_combi = hw_config_3d_combi(hw)
            hw_3d_sep = hw_config_3d_mem_on_logic(hw)
            for k in range(num_workload):
                if k == 0:
                    status = 'start'
                elif k == num_workload:
                    status = 'end'
                else:
                    status = 'middle'
                hw_2d.get_energy(workload[k], status, wbuf_dram)
                hw_3d_combi.get_energy_div_m(workload[k], status, wbuf_dram)
                hw_3d_combi.get_energy_div_c(workload[k], status, wbuf_dram)
                hw_3d_sep.get_energy(workload[k], status, wbuf_dram)
            hw_list.append([hw_2d, hw_3d_combi, hw_3d_sep])
            buf *= 4
        size *= 2
    im_list = np.zeros([8, 3])
    for p in range(5):        
        idx = p * 5 + p
        en_list = np.zeros([8, 3])
        for r in range(num_workload):
            tmp = workload[r].sa_2d_result[idx][4:9]
            en_list[0] += [tmp[0], tmp[1]+sum(tmp[2]), tmp[3]+tmp[4]]
            tmp = workload[r].dcim_2d_result[idx][4:9]
            en_list[1] += [tmp[0], tmp[1]+sum(tmp[2]), tmp[3]+tmp[4]]
            tmp = workload[r].sa_3d_divM_result[idx][4:9]
            en_list[2] += [tmp[0], tmp[1]+sum(tmp[2]), tmp[3]+tmp[4]]
            tmp = workload[r].dcim_3d_divM_result[idx][4:9]
            en_list[3] += [tmp[0], tmp[1]+sum(tmp[2]), tmp[3]+tmp[4]]
            tmp = workload[r].sa_3d_divC_result[idx][4:9]
            en_list[4] += [tmp[0], tmp[1]+sum(tmp[2]), tmp[3]+tmp[4]]
            tmp = workload[r].dcim_3d_divC_result[idx][4:9]
            en_list[5] += [tmp[0], tmp[1]+sum(tmp[2]), tmp[3]+tmp[4]]
            tmp = workload[r].sa_3d_result[idx][4:9]
            en_list[6] += [tmp[0], tmp[1]+sum(tmp[2]), tmp[3]+tmp[4]]
            tmp = workload[r].dcim_3d_result[idx][4:9]
            en_list[7] += [tmp[0], tmp[1]+sum(tmp[2]), tmp[3]+tmp[4]]
        en_list /= len(workload)
        result_arr[p] = np.array(en_list)

    en_improvement = np.sum(result_arr, axis=2)
    for d in range(5):
        en_improvement[d] /= en_improvement[d, 0]
    en_improvement = np.average(en_improvement, axis=0)
    en_improvement = np.reciprocal(en_improvement)
    tot_en += np.array([en_improvement[3], en_improvement[5], en_improvement[7]])


    buf_list = [64, 256, 1024, 4096, 16384]
    mac_list = [16**2, 32**2, 64**2, 128**2, 256**2]

    r2d_sa = result_arr[:, 0, :]
    r2d_cim = result_arr[:, 1, :]
    r3d_sa_divM = result_arr[:, 2, :]
    r3d_cim_divM = result_arr[:, 3, :]
    # r3d_sa_divC = result_arr[:, 4, :]
    # r3d_cim_divC = result_arr[:, 5, :]
    # r3d_sa = result_arr[:, 6, :]
    r3d_cim = result_arr[:, 7, :]
    

    # create data 
    x = [0, 2, 4, 6, 8] 
    width = 0.3
    # plot data in grouped manner of bar type 

    plt.xticks(x, ['0.25K', '1K', '4K', '16K', '64K']) 
    x_list = mac_list
    for x in range(5):
        bot1 = 0
        bot2 = 0
        bot3 = 0
        bot4 = 0
        bot5 = 0
        for i in range(3):  
            bar1 = plt.bar(2*x-2*width, r2d_sa[x, i], width, bottom=bot1, color=color[i], label='2D_SA_BL', edgecolor = 'black') 
            bar2 = plt.bar(2*x-width, r3d_sa_divM[x, i], width, bottom=bot2, color=color[i], label='3D_SA_1_divM', edgecolor = 'black') 
            bar3 = plt.bar(2*x+0.0, r2d_cim[x, i], width, bottom=bot3, color=color[i], label='2D_DCIM', edgecolor = 'black') 
            bar4 = plt.bar(2*x+width, r3d_cim_divM[x, i], width, bottom=bot4, color=color[i], label='3D_DCIM_1_divM', edgecolor = 'black') 
            bar5 = plt.bar(2*x+2*width, r3d_cim[x, i], width, bottom=bot5, color=color[i], label='3D_DCIM_2', edgecolor = 'black') 
            bot1 += r2d_sa[x, i]
            bot2 += r3d_sa_divM[x, i]
            bot3 += r2d_cim[x, i]
            bot4 += r3d_cim_divM[x, i]
            bot5 += r3d_cim[x, i]
            if x == 1 and i == 2 and z == 1:
                plt.annotate('(1)',xy=(2*x-2*width,bot1),xytext=(2*x-2*width ,bot1*1.15),  
                             arrowprops={'arrowstyle':'->'} ,horizontalalignment='right', fontsize=14, weight='bold')
                plt.annotate('(2)',xy=(2*x-width,bot2),xytext=(2*x-width+0.1 ,bot1*1.15),  
                             arrowprops={'arrowstyle':'->'} ,horizontalalignment='center', fontsize=14, weight='bold')
                plt.annotate('(3)',xy=(2*x,bot3),xytext=(2*x+0.5, bot1*1.15),  
                             arrowprops={'arrowstyle':'->'} ,horizontalalignment='center', fontsize=14, weight='bold')
                plt.annotate('(4)',xy=(2*x+width,bot4),xytext=(2*x+1.2 ,bot1*1.15),  
                             arrowprops={'arrowstyle':'->'} ,horizontalalignment='center', fontsize=14, weight='bold')
                plt.annotate('(5)',xy=(2*x+2*width,bot5),xytext=(2*x+1.9 ,bot1*1.15),  
                             arrowprops={'arrowstyle':'->'} ,horizontalalignment='center', fontsize=14, weight='bold')              
                    
    plt.title('{}'.format(wk_list[wk].strip('.csv')), weight='bold', size=18)
    plt.ylabel("Energy (\u03BCJ)", weight='bold', size=15)
    plt.xlabel("PE Array Size", weight='bold', size=15)
    plt.xticks(size=16, weight='bold')
    plt.yticks(size=16, weight='bold')
    ax = plt.gca()
    ax.tick_params(width=2)
    if z % 4 == 1:
        plt.legend(handles=[blue_patch, green_patch, yellow_patch], loc='upper center', prop={'weight':'bold', 'size': 15}, fontsize=15)
    z += 1

plt.show()

if args.s == True:
    plt.savefig('./Figures/Figure 7.png', dpi=300, bbox_inches='tight')