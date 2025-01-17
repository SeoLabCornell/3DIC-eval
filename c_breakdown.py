import numpy as np
import matplotlib.pyplot as plt
import csv
import matplotlib as mpl
import os
mpl.rcParams['lines.marker'] = '.'
plt.rcParams["font.family"] = "arial"
import argparse

from utils import *
from hardware import *
from workload import wk_layer
from carbon import *

parser = argparse.ArgumentParser()

parser.add_argument('-x', type=int, required=True,
                    help='[INT] x dim of compute array')
parser.add_argument('-y', type=int, required=True,
                    help='[INT] y dim of compute array')
parser.add_argument('--buf', type=int, required=True,
                    help='[INT] Total activation buffer size (KB)')

args = parser.parse_args()

base_path = './workloads/'
wk_list = os.listdir(base_path)
for i in range(len(wk_list)):
    wk_list[i] = wk_list[i].strip('.csv')

wk = wk_list.index('ResNet18')
workload_path = base_path + wk_list[wk] + '.csv'

packaging_intensity = 150 # gram CO2

sizex = args.x
sizey = args.y
buf = args.buf
buf_ratio = [1.2, 1.4, 1.6, 1.8, 2]

fig = plt.figure(figsize=(12,7))

workload = []

total_weight_num = 0
with open(workload_path, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    firstrow = next(reader)
    layer_num = 0
    for row in reader:
        workload.append(wk_layer(row, layer_num))
        layer_num += 1

num_workload = len(workload)
wbuf_dram = True
for lyr in range(num_workload):
    total_weight_num += workload[lyr].get_num_weights()

ax1 = plt.subplot(111)
wbuf = round(sizex*sizey/1024)*1024/16
hw = [sizex, sizey, 8, buf, buf, wbuf]
if total_weight_num <= wbuf * 1024:
    wbuf_dram = False
hw_2d = hw_config_2d(hw)
hw_3d_combi = hw_config_3d_combi(hw)
hw_3d_sep = hw_config_3d_mem_on_logic(hw)


debug = True

for i in range(num_workload):
    if i == 0:
        status = 'start'
    elif i == num_workload:
        status = 'end'
    else:
        status = 'middle'
    hw_2d.get_energy(workload[i], status, wbuf_dram)
    hw_3d_combi.get_energy_div_m(workload[i], status, wbuf_dram)
    hw_3d_sep.get_energy(workload[i], status, wbuf_dram)


idx = 0
en_list = np.zeros(6)
cycle_list = np.zeros(6)
for r in range(num_workload):
    en_list[0] += workload[r].sa_2d_result[idx][3]
    en_list[1] += workload[r].sa_3d_divM_result[idx][3]
    en_list[2] += workload[r].sa_3d_result[idx][3]
    en_list[3] += workload[r].dcim_2d_result[idx][3]        
    en_list[4] += workload[r].dcim_3d_divM_result[idx][3]         
    en_list[5] += workload[r].dcim_3d_result[idx][3]
    cycle_list[0] += workload[r].sa_2d_result[idx][1]
    cycle_list[1] += workload[r].sa_3d_divM_result[idx][1]
    cycle_list[2] += workload[r].sa_3d_result[idx][1]
    cycle_list[3] += workload[r].dcim_2d_result[idx][1]        
    cycle_list[4] += workload[r].dcim_3d_divM_result[idx][1]         
    cycle_list[5] += workload[r].dcim_3d_result[idx][1]        


xlist = ['2D_SA_BL'.format(round(sizex*sizey/1024)), 
            '3D_SA_1'.format(round(sizex*sizey/1024)), 
            '3D_SA_2'.format(round(sizex*sizey/1024)),
            '2D_DCIM'.format(round(sizex*sizey/1024)), 
            '3D_DCIM_1'.format(round(sizex*sizey/1024)),
            '3D_DCIM_2'.format(round(sizex*sizey/1024))]
ylist = ['Top Die', 'Bottom Die', 'Bonding', 'Package']

cde_linestyle = '-'
linewidth = 2
area_list = []
stacking = 'D2W'

print("----- 2D SA {}K -----".format(round(sizex*sizey/1024)))
tc_sa_2d = get_total_carbon(cal_carbon_hb(hw_2d.get_pe_total_area(), is_2d=True, stacking=stacking), is_2d=True)
area_list.append(hw_2d.get_pe_total_area())

print("\n----- 2D DCIM {}K -----".format(round(sizex*sizey/1024)))
tc_cim_2d = get_total_carbon(cal_carbon_hb(hw_2d.get_dcim_total_area(), stacking=stacking,is_2d=True), is_2d=True)
area_list.append(hw_3d_combi.get_pe_total_area())
area_list.append(hw_3d_sep.get_pe_total_area())
area_list.append(hw_2d.get_dcim_total_area())

print("\n----- 3D1 SA {}K -----".format(round(sizex*sizey/1024)))
tc_sa_3d1 = get_total_carbon(cal_carbon_hb(hw_3d_combi.get_pe_total_area(), stacking=stacking))

print("\n----- 3D1 DCIM {}K -----".format(round(sizex*sizey/1024)))
tc_cim_3d1 = get_total_carbon(cal_carbon_hb(hw_3d_combi.get_dcim_total_area(), stacking=stacking))
area_list.append(hw_3d_combi.get_dcim_total_area())

print("\n----- 3D2 SA {}K -----".format(round(sizex*sizey/1024)))
tc_sa_3d2 = get_total_carbon(cal_carbon_hb(hw_3d_sep.get_pe_total_area(), stacking=stacking))

print("\n----- 3D2 DCIM {}K -----".format(round(sizex*sizey/1024)))
tc_cim_3d2 = get_total_carbon(cal_carbon_hb(hw_3d_sep.get_dcim_total_area(), stacking=stacking))
area_list.append(hw_3d_sep.get_dcim_total_area())

data = np.array([tc_sa_2d,
                    tc_sa_3d1,
                    tc_sa_3d2,
                    tc_cim_2d,
                    tc_cim_3d1,
                    tc_cim_3d2])

left = np.zeros(6)
for i in range(4):
    color = ['b', 'g', 'c', 'r', 'm', 'mediumvioletred']
    bar = plt.barh(xlist, data[:,i], left=left, label=ylist[i], color=color[i], height=0.7)
    j = 0
    cl = 'w'
            
    left += data[:,i]

plt.xticks(weight='bold', size=18)
plt.yticks(np.arange(0, 6, step=1), weight='bold', size=18)
plt.legend(loc=(1.02,0.4), prop=dict(weight='bold', size=14))
plt.xlabel('Embodied Carbon (kg CO$_\mathbf{2}$)', weight='bold', size=20)

plt.show()