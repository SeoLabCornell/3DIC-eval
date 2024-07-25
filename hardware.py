import numpy as np
from utils import *

class hw_config_2d:
    def __init__(self, cur_hw):
        self.xdim = cur_hw[0]  # Systolic Array
        self.ydim = cur_hw[1]  # Systolic Array
        self.bit = cur_hw[2]

        self.ifbuf_size = cur_hw[3]  # KB
        self.ofbuf_size = cur_hw[4]  # KB
        self.wbuf_size = cur_hw[5]   # KB     
        
        self.pe_unit_area = 0.00121  # 0.00121mm2 Google TPU (ISCA 2017) Jouppi et al.
        self.pe_sram_area = get_sram_area(self.ifbuf_size) + get_sram_area(self.ofbuf_size) + get_sram_area(self.wbuf_size)
        self.pe_total_area = self.pe_unit_area * self.xdim * self.ydim + self.pe_sram_area
        
        self.mac_energy = get_mac_energy()  # 0.435 pJ/MAC Google TPU (ISCA 2017) Jouppi et al.
        self.dram_access_energy = 4.48 * self.bit  # 4.48pJ/bit Samsung LPDDR5 (JSSC 2020) Ha et al.
        self.ifbuf_energy = get_sram_energy(self.ifbuf_size)  # pJ/access
        self.ofbuf_energy = get_sram_energy(self.ofbuf_size)  # pJ/access
        self.wbuf_energy = get_sram_energy(self.wbuf_size)    # pJ/access
        self.wire_energy = 0.2  # 0.2pJ/bit/mm
        
        self.buf_to_pe_wire = np.sqrt(self.pe_unit_area * self.xdim * self.ydim) / 2
        self.buf_to_dram_wire = np.sqrt(self.pe_total_area) / 2
        self.set_dcim_attrs()

    def get_num_mac(self):
        return self.xdim * self.ydim

    def get_pe_total_area(self):
        return self.pe_total_area

    def get_dcim_total_area(self):
        return self.dcim_total_area

    def get_dcim_dim(self):
        return [self.xdim_dcim, self.ydim_dcim]

    def set_dcim_attrs(self):
        self.num_dcim_macro = self.get_num_mac() / 32  # DCIM Max. Thruput 256/Cycle (ESSCIRC 2023) Oh et al.
        self.xdim_dcim = np.round(np.sqrt(self.num_dcim_macro))
        self.ydim_dcim = np.round(np.sqrt(self.num_dcim_macro))
        self.dcim_mac_energy = 0.08  # 0.0893pJ/MAC (ESSCIRC 2023) Oh et al.
        
        self.dcim_unit_area = 0.0159 * 1  # 0.0159mm2 (ESSCIRC 2023) Oh et al.
        self.dcim_sram_area = get_sram_area(self.ifbuf_size) + get_sram_area(self.ofbuf_size)
        self.dcim_total_area = self.dcim_unit_area * self.xdim_dcim * self.ydim_dcim + self.dcim_sram_area       

        self.buf_to_dcim_wire = np.sqrt(self.dcim_unit_area * self.xdim_dcim * self.ydim_dcim) / 2
        self.buf_to_dram_wire_dcim = np.sqrt(self.dcim_total_area) / 2

    def get_energy(self, wk_layer, status, wbuf_dram):
        # 1. Get cycles and util
        num_mac_op = np.ceil(wk_layer.get_num_computes() / self.get_num_mac())
        
        row_fold = np.ceil(wk_layer.filter_H * wk_layer.filter_W * wk_layer.num_channel / self.ydim)  # Sr/R
        row_mod = np.mod(wk_layer.filter_H * wk_layer.filter_W * wk_layer.num_channel, self.ydim)  # Row modulus
        if row_mod == 0:
            row_mod = self.ydim
        col_fold = np.ceil(wk_layer.num_filter / self.xdim)  # Sc/C
        col_mod = np.mod(wk_layer.num_filter, self.xdim)  # Column modulus
        if col_mod == 0:
            col_mod = self.xdim
        T = wk_layer.ofmap_H * wk_layer.ofmap_W  # T (temporal size)
        cycle_per_fold = 2 * self.xdim + self.ydim + T - 2  # Cycles per one pair of fold

        cycles = row_fold * col_fold * cycle_per_fold  # Total number of cycles
        util = num_mac_op / cycles  # Utilization
              
        # 2-a. Get PE energy
        total_pe_energy = num_mac_op * self.get_num_mac() * self.mac_energy / 1e6  # uJ
        # 2-b. Get DCIM energy
        total_dcim_energy = num_mac_op * self.get_num_mac() * self.dcim_mac_energy / 1e6  # uJ

        # 3. Get Buffer Accesses
        wbuf_read = wk_layer.get_num_weights()
        wbuf_fill = wk_layer.get_num_weights()
        if not wbuf_dram:
            wbuf_fill = 0
        ifbuf_read = col_fold * T * (self.ydim * row_fold - (self.ydim - row_mod))
        if self.ifbuf_size * 8 * 1024 / (self.bit) >= wk_layer.get_num_inputs():
            if status == 'start':
                ifbuf_fill = wk_layer.get_num_inputs()
            else:
                ifbuf_fill = 0
        else:
            if status == 'start':
                ifbuf_fill = self.ifbuf_size * 1024 + (wk_layer.get_num_inputs() - self.ifbuf_size * 1024) * col_fold
            else:
                ifbuf_fill = (wk_layer.get_num_inputs() - self.ifbuf_size * 1024) * col_fold
        ofbuf_update = row_fold * T * (self.xdim * col_fold - (self.xdim - col_mod))
        if status == 'last':
            ofbuf_fill = wk_layer.get_num_outputs()
        else:
            if self.ofbuf_size * 8 * 1024 / (self.bit) >= wk_layer.get_num_outputs():
                ofbuf_fill = 0
            else:
                ofbuf_fill = wk_layer.get_num_outputs() - self.ofbuf_size * 1024

        # 4. Get Buffer Energy
        wbuf_energy = (wbuf_read + wbuf_fill) * self.wbuf_energy / 1e6  # uJ
        ifbuf_energy = (ifbuf_read + ifbuf_fill) * self.ifbuf_energy / 1e6  # uJ
        ofbuf_energy = (ofbuf_update + ofbuf_fill) * self.ofbuf_energy / 1e6  # uJ

        # 5. Get DRAM Energy
        dram_energy = (wbuf_fill + ifbuf_fill + ofbuf_fill) * self.dram_access_energy / 1e6  # uJ

        # 6-a. Get Wire Energy (Systolic Array)
        wire_buf_pe_energy = (wbuf_read + ifbuf_read + ofbuf_update) * self.buf_to_pe_wire * self.wire_energy * self.bit / 1e6  # uJ
        wire_buf_dram_energy = (wbuf_fill + ifbuf_fill + ofbuf_fill) * self.buf_to_dram_wire * self.wire_energy * self.bit / 1e6  # uJ
        # 6-b. Get Wire Energy (DCIM)
        wire_buf_pe_energy_dcim = (wbuf_read + ifbuf_read + ofbuf_update) * self.buf_to_dcim_wire * self.wire_energy * self.bit / 1e6  # uJ
        wire_buf_dram_energy_dcim = (wbuf_fill + ifbuf_fill + ofbuf_fill) * self.buf_to_dram_wire_dcim * self.wire_energy * self.bit / 1e6  # uJ

        # 7-a. Gather Energy (Systolic Array)
        sys_buffer_energy = [wbuf_energy, ifbuf_energy, ofbuf_energy]
        sys_wire_energy = [wire_buf_pe_energy, wire_buf_dram_energy]
        sys_total_energy = sum(sys_buffer_energy) + sum(sys_wire_energy) + dram_energy + total_pe_energy
        # 7-b. Gather Energy (DCIM)
        dcim_buffer_energy = [0, ifbuf_energy, ofbuf_energy]
        dcim_wire_energy = [wire_buf_pe_energy_dcim, wire_buf_dram_energy_dcim]
        dcim_total_energy = sum(dcim_buffer_energy) + sum(dcim_wire_energy) + dram_energy + total_dcim_energy

        # 8. Gather Buffer Accesses
        buf_access = [wbuf_read, ifbuf_read, ofbuf_update, wbuf_fill, ifbuf_fill, ofbuf_fill]

        sys_result = ['2d_systolic', cycles, util, sys_total_energy, total_pe_energy, sys_wire_energy[0], sys_buffer_energy, sys_wire_energy[1], dram_energy, buf_access, self.pe_total_area, num_mac_op]
        dcim_result = ['2d_dcim', cycles, util, dcim_total_energy, total_dcim_energy, dcim_wire_energy[0], dcim_buffer_energy, dcim_wire_energy[1], dram_energy, buf_access, self.dcim_total_area, num_mac_op]
        wk_layer.save_sim_results(sys_result)
        wk_layer.save_sim_results(dcim_result)
    

class hw_config_3d_combi:
    def __init__(self, cur_hw):
        self.xdim_top = get_3d_array_size(cur_hw[0])  # Systolic Array
        self.ydim_top = get_3d_array_size(cur_hw[1])  # Systolic Array
        self.xdim_bot = self.xdim_top
        self.ydim_bot = self.ydim_top
        self.bit = cur_hw[2]

        self.ifbuf_size_top = cur_hw[3] / 2  # KB
        self.ofbuf_size_top = cur_hw[4] / 2  # KB
        self.wbuf_size_top = cur_hw[5] / 2   # KB
        self.ifbuf_size_bot = cur_hw[3] / 2  # KB
        self.ofbuf_size_bot = cur_hw[4] / 2  # KB
        self.wbuf_size_bot = cur_hw[5] / 2   # KB        
        
        self.pe_unit_area = 0.00121  # 0.00121mm2 Google TPU (ISCA 2017) Jouppi et al.
        self.pe_sram_area = get_sram_area(self.ifbuf_size_bot) + get_sram_area(self.ofbuf_size_bot) + get_sram_area(self.wbuf_size_bot)
        self.pe_total_area = self.pe_unit_area * self.xdim_bot * self.ydim_bot + self.pe_sram_area
        
        self.mac_energy = get_mac_energy()  
        self.dram_access_energy = 4.48 * self.bit  # 4.48pJ/bit Samsung LPDDR5 (JSSC 2020) Ha et al.
        self.ifbuf_energy = get_sram_energy(self.ifbuf_size_bot)  # pJ/access
        self.ofbuf_energy = get_sram_energy(self.ofbuf_size_bot)  # pJ/access
        self.wbuf_energy = get_sram_energy(self.wbuf_size_bot)    # pJ/access
        self.wire_energy = 0.2  # 0.2pJ/bit/mm
        self.hb_energy = 0.013 * self.bit  # Hybrid Bonding 0.013pJ/bit S. Sinha et al. 2020 IEDM
        
        self.buf_to_pe_wire = np.sqrt(self.pe_unit_area * self.xdim_bot * self.ydim_bot) / 2
        self.buf_to_dram_wire = np.sqrt(self.pe_total_area) / 2
        self.set_dcim_attrs()

    def get_num_mac(self):
        return self.xdim_bot * self.ydim_bot + self.xdim_top * self.ydim_top

    def get_pe_total_area(self):
        return self.pe_total_area

    def get_dcim_total_area(self):
        return self.dcim_total_area

    def get_dcim_dim(self):
        return [self.xdim_dcim_bot, self.ydim_dcim_bot]

    def set_dcim_attrs(self):
        self.num_dcim_macro_bot = np.ceil(0.5 * self.get_num_mac() / 32)  # DCIM Max. Thruput 256/Cycle (ESSCIRC 2023) Oh et al.
        self.num_dcim_macro_top = self.num_dcim_macro_bot
        self.xdim_dcim_bot = np.round(np.sqrt(self.num_dcim_macro_bot))
        self.ydim_dcim_bot = np.round(np.sqrt(self.num_dcim_macro_bot))
        self.xdim_dcim_top = self.xdim_dcim_bot
        self.ydim_dcim_top = self.ydim_dcim_bot
        self.dcim_mac_energy = 0.08  # 0.0532pJ/MAC (ESSCIRC 2023) Oh et al.
        
        self.dcim_unit_area = 0.0159 * 1  # 0.0159mm2 (ESSCIRC 2023) Oh et al.
        self.dcim_sram_area = get_sram_area(self.ifbuf_size_bot) + get_sram_area(self.ofbuf_size_bot)
        self.dcim_total_area = self.dcim_unit_area * self.num_dcim_macro_bot + self.dcim_sram_area      

        self.buf_to_dcim_wire = np.sqrt(self.dcim_unit_area * self.num_dcim_macro_bot) / 2
        self.buf_to_dram_wire_dcim = np.sqrt(self.dcim_total_area) / 2

    def get_energy_div_m(self, wk_layer, status, wbuf_dram):
        # This means for each die, num_weight is mapped on vertical dimension (say M/2)
        # so, each die will share same set of inputs for every cycle, while different set of weights
        # 1. Get cycles and util
        num_mac_op = np.ceil(wk_layer.get_num_computes() / self.get_num_mac())
        
        row_fold = np.ceil(wk_layer.filter_H * wk_layer.filter_W * wk_layer.num_channel / self.ydim_bot)  # Sr/R
        row_mod = np.mod(wk_layer.filter_H * wk_layer.filter_W * wk_layer.num_channel, self.ydim_bot)  # Row modulus
        if row_mod == 0:
            row_mod = self.ydim_bot
        col_fold = np.ceil(0.5 * wk_layer.num_filter / self.xdim_bot)  # Sc/C
        col_mod = np.mod(0.5 * wk_layer.num_filter, self.xdim_bot)  # Column modulus
        if col_mod == 0:
            col_mod = self.xdim_bot
        T = wk_layer.ofmap_H * wk_layer.ofmap_W  # T (temporal size)
        cycle_per_fold = 2 * self.xdim_bot + self.ydim_bot + T - 2  # Cycles per one pair of fold

        cycles = row_fold * col_fold * cycle_per_fold  # Total number of cycles
        util = num_mac_op / cycles  # Utilization
              
        # 2-a. Get PE energy
        total_pe_energy = num_mac_op * self.get_num_mac() * self.mac_energy / 1e6  # uJ
        # 2-b. Get DCIM energy
        num_dcim_mac_op = (0.5 * wk_layer.get_num_computes()) / (256 * self.num_dcim_macro_bot)
        total_dcim_energy = num_dcim_mac_op * (self.num_dcim_macro_top + self.num_dcim_macro_bot) * 256 * self.dcim_mac_energy / 1e6  # uJ

        # 3. Get Buffer Accesses
        wbuf_read_top = wk_layer.get_num_weights() / 2
        wbuf_fill_top = wk_layer.get_num_weights() / 2
        if not wbuf_dram:
            wbuf_fill_top = 0
        wbuf_read_bot = wk_layer.get_num_weights() / 2
        wbuf_fill_bot = wk_layer.get_num_weights() / 2
        if not wbuf_dram:
            wbuf_fill_bot = 0
        ifbuf_read_top = 0  # Because the inputs are shared from bottom
        ifbuf_fill_top = 0
        ifbuf_read_bot = col_fold * T * (self.ydim_bot * row_fold - (self.ydim_bot - row_mod))
        if self.ifbuf_size_bot * 8 * 1024 * 2 / self.bit >= wk_layer.get_num_inputs():
            if status == 'start':
                ifbuf_fill_bot = wk_layer.get_num_inputs()
            else:
                ifbuf_fill_bot = 0
        else:
            if status == 'start':
                ifbuf_fill_bot = self.ifbuf_size_bot * 1024 * 2 + (wk_layer.get_num_inputs() - self.ifbuf_size_bot * 1024 * 2) * col_fold
            else:
                ifbuf_fill_bot = (wk_layer.get_num_inputs() - self.ifbuf_size_bot * 1024 * 2) * col_fold
        ofbuf_update_top = row_fold * T * (self.xdim_top * col_fold - (self.xdim_top - col_mod))
        ofbuf_update_bot = row_fold * T * (self.xdim_bot * col_fold - (self.xdim_bot - col_mod))
        if status == 'last':
            ofbuf_fill_top = wk_layer.get_num_outputs() / 2
            ofbuf_fill_bot = wk_layer.get_num_outputs() / 2
        else:
            if self.ofbuf_size_bot * 8 * 1024 / (self.bit) >= wk_layer.get_num_outputs() / 2:
                ofbuf_fill_bot = 0
                ofbuf_fill_top = 0
            else:
                ofbuf_fill_bot = (wk_layer.get_num_outputs() / 2) - self.ofbuf_size_bot * 1024
                ofbuf_fill_top = (wk_layer.get_num_outputs() / 2) - self.ofbuf_size_top * 1024
        
        # 4. Get Buffer Energy
        wbuf_energy_top = (wbuf_read_top + wbuf_fill_top) * self.wbuf_energy / 1e6  # uJ
        ifbuf_energy_top = (ifbuf_read_top + ifbuf_fill_top) * self.ifbuf_energy / 1e6  # uJ
        ofbuf_energy_top = (ofbuf_update_top + ofbuf_fill_top) * self.ofbuf_energy / 1e6  # uJ
        wbuf_energy_bot = (wbuf_read_bot + wbuf_fill_bot) * self.wbuf_energy / 1e6  # uJ
        ifbuf_energy_bot = (ifbuf_read_bot + ifbuf_fill_bot) * self.ifbuf_energy / 1e6  # uJ
        ofbuf_energy_bot = (ofbuf_update_bot + ofbuf_fill_bot) * self.ofbuf_energy / 1e6  # uJ

        # 5. Get DRAM Energy
        dram_energy = (wbuf_fill_top + ifbuf_fill_top + ofbuf_fill_top + wbuf_fill_bot + ifbuf_fill_bot + ofbuf_fill_bot) * self.dram_access_energy / 1e6  # uJ

        # 6-a. Get Wire Energy (Systolic Array)
        ## 2D Wire Energy
        wire_buf_pe_energy_top = (wbuf_read_top + ifbuf_read_top + ofbuf_update_top) * self.buf_to_pe_wire * self.wire_energy * self.bit / 1e6  # uJ
        wire_buf_dram_energy_top = (wbuf_fill_top + ifbuf_fill_top + ofbuf_fill_top) * self.buf_to_dram_wire * self.wire_energy * self.bit / 1e6  # uJ
        wire_buf_pe_energy_bot = (wbuf_read_bot + ifbuf_read_bot + ofbuf_update_bot) * self.buf_to_pe_wire * self.wire_energy * self.bit / 1e6  # uJ
        wire_buf_dram_energy_bot = (wbuf_fill_bot + ifbuf_fill_bot + ofbuf_fill_bot) * self.buf_to_dram_wire * self.wire_energy * self.bit / 1e6  # uJ
        # 6-b. Get Wire Energy (DCIM)
        wire_buf_pe_energy_top_dcim = (wbuf_read_top + ifbuf_read_top + ofbuf_update_top) * self.buf_to_dcim_wire * self.wire_energy * self.bit / 1e6  # uJ
        wire_buf_dram_energy_top_dcim = (wbuf_fill_top + ifbuf_fill_top + ofbuf_fill_top) * self.buf_to_dram_wire_dcim * self.wire_energy * self.bit / 1e6  # uJ
        wire_buf_pe_energy_bot_dcim = (wbuf_read_bot + ifbuf_read_bot + ofbuf_update_bot) * self.buf_to_dcim_wire * self.wire_energy * self.bit / 1e6  # uJ
        wire_buf_dram_energy_bot_dcim = (wbuf_fill_bot + ifbuf_fill_bot + ofbuf_fill_bot) * self.buf_to_dram_wire_dcim * self.wire_energy * self.bit / 1e6  # uJ

        # 7. 3D Wire Energy
        hb_energy = (ifbuf_read_bot + wbuf_fill_top + ofbuf_fill_top) * self.hb_energy / 1e6
        hb_energy_1 = ifbuf_read_bot * self.hb_energy / 1e6
        hb_energy_2 = (wbuf_fill_top + ofbuf_fill_top) * self.hb_energy / 1e6

        # 8-a. Gather Energy (Systolic Array)
        sys_buffer_energy = [wbuf_energy_top, ifbuf_energy_top, ofbuf_energy_top, wbuf_energy_bot, ifbuf_energy_bot, ofbuf_energy_bot]
        sys_wire_energy = [wire_buf_pe_energy_top, wire_buf_dram_energy_top, wire_buf_pe_energy_bot, wire_buf_dram_energy_bot, hb_energy]
        sys_total_energy = sum(sys_buffer_energy) + sum(sys_wire_energy) + dram_energy + total_pe_energy
        # 8-b. Gather Energy (DCIM)
        dcim_buffer_energy = [0, ifbuf_energy_top, ofbuf_energy_top, 0, ifbuf_energy_bot, ofbuf_energy_bot]
        dcim_wire_energy = [wire_buf_pe_energy_top_dcim, wire_buf_dram_energy_top_dcim, wire_buf_pe_energy_bot_dcim, wire_buf_dram_energy_bot_dcim, hb_energy]
        dcim_total_energy = sum(dcim_buffer_energy) + sum(dcim_wire_energy) + dram_energy + total_dcim_energy

        # 9. Gather Buffer Accesses
        buf_access_top = [wbuf_read_top, ifbuf_read_top, ofbuf_update_top, wbuf_fill_top, ifbuf_fill_top, ofbuf_fill_top]
        buf_access_bot = [wbuf_read_bot, ifbuf_read_bot, ofbuf_update_bot, wbuf_fill_bot, ifbuf_fill_bot, ofbuf_fill_bot]
        buf_access = [sum(x) for x in zip(buf_access_top, buf_access_bot)]

        sys_result = ['3d_systolic_combi_div_m', cycles, util, sys_total_energy, total_pe_energy, sys_wire_energy[0]+sys_wire_energy[2]+hb_energy_1, 
                      sys_buffer_energy, sys_wire_energy[1]+sys_wire_energy[3]+hb_energy_2, dram_energy, sys_wire_energy[4], 
                      buf_access, self.pe_total_area, num_mac_op*2]
        dcim_result = ['3d_dcim_combi_div_m', cycles, util, dcim_total_energy, total_dcim_energy, dcim_wire_energy[0]+dcim_wire_energy[2]+hb_energy_1, 
                       dcim_buffer_energy, dcim_wire_energy[1]+dcim_wire_energy[3]+hb_energy_2, dram_energy, sys_wire_energy[4],
                       buf_access, self.dcim_total_area, num_mac_op*2]
        wk_layer.save_sim_results(sys_result)
        wk_layer.save_sim_results(dcim_result)
        
    def get_energy_div_c(self, wk_layer, status, wbuf_dram):
        # This means for each die, channel is mapped on vertical dimension (say C/2)
        # so, each die will share half set of inputs and weights for every cycle
        # At the bottom of each die, psums from top die will move down and gets addeded (C/2 + C/2 = C)
        # 1. Get cycles and util
        num_mac_op = np.ceil(wk_layer.get_num_computes() / self.get_num_mac())
        half_c = 0.5 * wk_layer.num_channel
        
        row_fold = np.ceil(wk_layer.filter_H * wk_layer.filter_W * half_c / self.ydim_bot)  # Sr/R
        row_mod = np.mod(wk_layer.filter_H * wk_layer.filter_W * half_c, self.ydim_bot)  # Row modulus
        if row_mod == 0:
            row_mod = self.ydim_bot
        col_fold = np.ceil(wk_layer.num_filter / self.xdim_bot)  # Sc/C
        col_mod = np.mod(wk_layer.num_filter, self.xdim_bot)  # Column modulus
        if col_mod == 0:
            col_mod = self.xdim_bot
        T = wk_layer.ofmap_H * wk_layer.ofmap_W  # T (temporal size)
        cycle_per_fold = 2 * self.xdim_bot + self.ydim_bot + T - 2  # Cycles per one pair of fold

        cycles = row_fold * col_fold * cycle_per_fold  # Total number of cycles
        util = num_mac_op / cycles  # Utilization
              
        # 2-a. Get PE energy
        total_pe_energy = num_mac_op * self.get_num_mac() * self.mac_energy / 1e6  # uJ
        # 2-b. Get DCIM energy
        num_dcim_mac_op = (0.5 * wk_layer.get_num_computes()) / (256 * self.num_dcim_macro_bot)
        total_dcim_energy = num_dcim_mac_op * (self.num_dcim_macro_top + self.num_dcim_macro_bot) * 256 * self.dcim_mac_energy / 1e6  # uJ

        # 3. Get Buffer Accesses
        wbuf_read_top = wk_layer.get_num_weights() / 2
        wbuf_fill_top = wk_layer.get_num_weights() / 2
        wbuf_read_bot = wk_layer.get_num_weights() / 2
        wbuf_fill_bot = wk_layer.get_num_weights() / 2
        if not wbuf_dram:
            wbuf_fill_top = 0
            wbuf_fill_bot = 0
        ifbuf_read_bot = col_fold * T * (self.ydim_bot * row_fold - (self.ydim_bot - row_mod))
        if self.ifbuf_size_bot * 8 * 1024  / self.bit >= wk_layer.get_num_inputs() / 2:
            if status == 'start':
                ifbuf_fill_bot = wk_layer.get_num_inputs() * 0.5
            else:
                ifbuf_fill_bot = 0
                
        else:
            if status == 'start':
                ifbuf_fill_bot = self.ifbuf_size_bot * 1024 + (0.5*wk_layer.get_num_inputs() - self.ifbuf_size_bot * 1024) * col_fold
            else:
                ifbuf_fill_bot = (0.5*wk_layer.get_num_inputs() - self.ifbuf_size_bot * 1024) * col_fold            
        ifbuf_read_top = ifbuf_read_bot
        ifbuf_fill_top = ifbuf_fill_bot
        ofbuf_update_top = 0  # Because the psums are temporally diminished vertically
        ofbuf_fill_top = 0
        ofbuf_update_bot = row_fold * T * (self.xdim_bot * col_fold - (self.xdim_bot - col_mod))
        if status == 'last':
            ofbuf_fill_bot = wk_layer.get_num_outputs()
        else:
            if self.ofbuf_size_bot * 8 * 1024 * 2 / self.bit >= wk_layer.get_num_outputs():
                ofbuf_fill_bot = 0
            else:
                ofbuf_fill_bot = wk_layer.get_num_outputs() - self.ofbuf_size_bot * 1024 * 2

        
        # 4. Get Buffer Energy
        wbuf_energy_top = (wbuf_read_top + wbuf_fill_top) * self.wbuf_energy / 1e6  # uJ
        ifbuf_energy_top = (ifbuf_read_top + ifbuf_fill_top) * self.ifbuf_energy / 1e6  # uJ
        ofbuf_energy_top = (ofbuf_update_top + ofbuf_fill_top) * self.ofbuf_energy / 1e6  # uJ
        wbuf_energy_bot = (wbuf_read_bot + wbuf_fill_bot) * self.wbuf_energy / 1e6  # uJ
        ifbuf_energy_bot = (ifbuf_read_bot + ifbuf_fill_bot) * self.ifbuf_energy / 1e6  # uJ
        ofbuf_energy_bot = (ofbuf_update_bot + ofbuf_fill_bot) * self.ofbuf_energy / 1e6  # uJ

        # 5. Get DRAM Energy
        dram_energy = (wbuf_fill_top + ifbuf_fill_top + ofbuf_fill_top + wbuf_fill_bot + ifbuf_fill_bot + ofbuf_fill_bot) * self.dram_access_energy / 1e6  # uJ

        # 6-a. Get Wire Energy (Systolic Array)
        ## 2D Wire Energy
        wire_buf_pe_energy_top = (wbuf_read_top + ifbuf_read_top + ofbuf_update_top) * self.buf_to_pe_wire * self.wire_energy * self.bit / 1e6  # uJ
        wire_buf_dram_energy_top = (wbuf_fill_top + ifbuf_fill_top + ofbuf_fill_top) * self.buf_to_dram_wire * self.wire_energy * self.bit / 1e6  # uJ
        wire_buf_pe_energy_bot = (wbuf_read_bot + ifbuf_read_bot + ofbuf_update_bot) * self.buf_to_pe_wire * self.wire_energy * self.bit / 1e6  # uJ
        wire_buf_dram_energy_bot = (wbuf_fill_bot + ifbuf_fill_bot + ofbuf_fill_bot) * self.buf_to_dram_wire * self.wire_energy * self.bit / 1e6  # uJ
        # 6-b. Get Wire Energy (DCIM)
        wire_buf_pe_energy_top_dcim = (wbuf_read_top + ifbuf_read_top + ofbuf_update_top) * self.buf_to_dcim_wire * self.wire_energy * self.bit / 1e6  # uJ
        wire_buf_dram_energy_top_dcim = (wbuf_fill_top + ifbuf_fill_top + ofbuf_fill_top) * self.buf_to_dram_wire_dcim * self.wire_energy * self.bit / 1e6  # uJ
        wire_buf_pe_energy_bot_dcim = (wbuf_read_bot + ifbuf_read_bot + ofbuf_update_bot) * self.buf_to_dcim_wire * self.wire_energy * self.bit / 1e6  # uJ
        wire_buf_dram_energy_bot_dcim = (wbuf_fill_bot + ifbuf_fill_bot + ofbuf_fill_bot) * self.buf_to_dram_wire_dcim * self.wire_energy * self.bit / 1e6  # uJ

        # 7. 3D Wire Energy
        hb_energy = (ofbuf_update_bot + wbuf_fill_top + ifbuf_fill_top) * self.hb_energy / 1e6
        hb_energy_1 = ofbuf_update_bot * self.hb_energy / 1e6
        hb_energy_2 = (wbuf_fill_top + ifbuf_fill_top) * self.hb_energy / 1e6

        # 8-a. Gather Energy (Systolic Array)
        sys_buffer_energy = [wbuf_energy_top, ifbuf_energy_top, ofbuf_energy_top, wbuf_energy_bot, ifbuf_energy_bot, ofbuf_energy_bot]
        sys_wire_energy = [wire_buf_pe_energy_top, wire_buf_dram_energy_top, wire_buf_pe_energy_bot, wire_buf_dram_energy_bot, hb_energy]
        sys_total_energy = sum(sys_buffer_energy) + sum(sys_wire_energy) + dram_energy + total_pe_energy
        # 8-b. Gather Energy (DCIM)
        dcim_buffer_energy = [0, ifbuf_energy_top, ofbuf_energy_top, 0, ifbuf_energy_bot, ofbuf_energy_bot]
        dcim_wire_energy = [wire_buf_pe_energy_top_dcim, wire_buf_dram_energy_top_dcim, wire_buf_pe_energy_bot_dcim, wire_buf_dram_energy_bot_dcim, hb_energy]
        dcim_total_energy = sum(dcim_buffer_energy) + sum(dcim_wire_energy) + dram_energy + total_dcim_energy

        # 9. Gather Buffer Accesses
        buf_access_top = [wbuf_read_top, ifbuf_read_top, ofbuf_update_top, wbuf_fill_top, ifbuf_fill_top, ofbuf_fill_top]
        buf_access_bot = [wbuf_read_bot, ifbuf_read_bot, ofbuf_update_bot, wbuf_fill_bot, ifbuf_fill_bot, ofbuf_fill_bot]
        buf_access = [sum(x) for x in zip(buf_access_top, buf_access_bot)]

        sys_result = ['3d_systolic_combi_div_c', cycles, util, sys_total_energy, total_pe_energy, sys_wire_energy[0]+sys_wire_energy[2]+hb_energy_1,
                      sys_buffer_energy, sys_wire_energy[1]+sys_wire_energy[3]+hb_energy_2, dram_energy, sys_wire_energy[4],
                      buf_access, self.pe_total_area, num_mac_op*2]
        dcim_result = ['3d_dcim_combi_div_c', cycles, util, dcim_total_energy, total_dcim_energy, dcim_wire_energy[0]+dcim_wire_energy[2]+hb_energy_1, 
                       dcim_buffer_energy, dcim_wire_energy[1]+dcim_wire_energy[3]+hb_energy_2, dram_energy, dcim_wire_energy[4],
                       buf_access, self.dcim_total_area, num_mac_op*2]
        wk_layer.save_sim_results(sys_result)
        wk_layer.save_sim_results(dcim_result)

class hw_config_3d_mem_on_logic:
    def __init__(self, cur_hw):
        self.xdim = cur_hw[0]   # Systolic Array
        self.ydim = cur_hw[1]   # Systolic Array
        self.bit = cur_hw[2]

        self.ifbuf_size = cur_hw[3]  # KB
        self.ofbuf_size = cur_hw[4]  # KB
        self.wbuf_size = cur_hw[5]   # KB

        self.pe_ofbuf_bot = False
        
        self.pe_unit_area = 0.00121  # 0.00121mm2 Google TPU (ISCA 2017) Jouppi et al.
        self.pe_sram_area = get_sram_area(self.ifbuf_size) + get_sram_area(self.ofbuf_size)
        self.pe_total_area = self.pe_unit_area * self.xdim * self.ydim
        if self.pe_total_area < self.pe_sram_area:
            self.pe_total_area = self.pe_sram_area

        if self.pe_total_area > self.pe_unit_area * self.xdim * self.ydim + get_sram_area(self.ofbuf_size):
            self.pe_ofbuf_bot = True
            self.pe_total_area = self.pe_unit_area * self.xdim * self.ydim + get_sram_area(self.ofbuf_size)
        
        self.mac_energy = get_mac_energy()  
        self.dram_access_energy = 4.48 * self.bit  # 4.48pJ/bit Samsung LPDDR5 (JSSC 2020) Ha et al.
        self.ifbuf_energy = get_sram_energy(self.ifbuf_size)  # pJ/access
        self.ofbuf_energy = get_sram_energy(self.ofbuf_size)  # pJ/access
        self.wbuf_energy = get_sram_energy(self.wbuf_size)    # pJ/access
        self.wire_energy = 0.2  # 0.2pJ/bit/mm
        self.hb_energy = 0.013 * self.bit  # Hybrid Bonding 0.013pJ/bit S. Sinha et al. 2020 IEDM
        
        self.buf_to_pe_wire = np.sqrt(self.pe_unit_area * self.xdim * self.ydim) / 2
        self.buf_to_dram_wire = np.sqrt(self.pe_total_area) / 2
        self.set_dcim_attrs()

    def get_num_mac(self):
        return self.xdim * self.ydim

    def get_pe_total_area(self):
        return self.pe_total_area

    def get_dcim_total_area(self):
        return self.dcim_total_area

    def get_dcim_dim(self):
        return [self.xdim_dcim, self.ydim_dcim]

    def set_dcim_attrs(self):
        self.num_dcim_macro = self.get_num_mac() / 32  # DCIM Max. Thruput 256/Cycle (ESSCIRC 2023) Oh et al.
        self.xdim_dcim = np.round(np.sqrt(self.num_dcim_macro))
        self.ydim_dcim = np.round(np.sqrt(self.num_dcim_macro))
        self.dcim_mac_energy = 0.08  # 0.0893pJ/MAC (ESSCIRC 2023) Oh et al.

        self.dcim_ofbuf_bot = False
        
        self.dcim_unit_area = 0.0159 * 1 # 0.0159mm2 (ESSCIRC 2023) Oh et al.
        self.dcim_sram_area = get_sram_area(self.ifbuf_size) + get_sram_area(self.ofbuf_size)
        self.dcim_total_area = self.dcim_unit_area * self.xdim_dcim * self.ydim_dcim
        if self.dcim_total_area < self.dcim_sram_area:
            self.dcim_total_area = self.dcim_sram_area

        if self.dcim_total_area > self.dcim_unit_area * self.xdim_dcim * self.ydim_dcim + get_sram_area(self.ofbuf_size):
            self.dcim_ofbuf_bot = True
            self.dcim_total_area = self.dcim_unit_area * self.xdim_dcim * self.ydim_dcim + get_sram_area(self.ofbuf_size)

        self.buf_to_dcim_wire = np.sqrt(self.dcim_unit_area * self.xdim_dcim * self.ydim_dcim) / 2
        self.buf_to_dram_wire_dcim = np.sqrt(self.dcim_total_area) / 2

    def get_energy(self, wk_layer, status, wbuf_dram):
        # 1. Get cycles and util
        num_mac_op = np.ceil(wk_layer.get_num_computes() / self.get_num_mac())
        
        row_fold = np.ceil(wk_layer.filter_H * wk_layer.filter_W * wk_layer.num_channel / self.ydim)  # Sr/R
        row_mod = np.mod(wk_layer.filter_H * wk_layer.filter_W * wk_layer.num_channel, self.ydim)  # Row modulus
        if row_mod == 0:
            row_mod = self.ydim
        col_fold = np.ceil(wk_layer.num_filter / self.xdim)  # Sc/C
        col_mod = np.mod(wk_layer.num_filter, self.xdim)  # Column modulus
        if col_mod == 0:
            col_mod = self.xdim
        T = wk_layer.ofmap_H * wk_layer.ofmap_W  # T (temporal size)
        cycle_per_fold = self.xdim + self.ydim + T - 1  # Cycles per one pair of fold

        cycles = row_fold * col_fold * cycle_per_fold  # Total number of cycles
        util = num_mac_op / cycles  # Utilization
              
        # 2-a. Get PE energy
        total_pe_energy = num_mac_op * self.get_num_mac() * self.mac_energy / 1e6  # uJ
        # 2-b. Get DCIM energy
        total_dcim_energy = num_mac_op * self.get_num_mac() * self.dcim_mac_energy / 1e6  # uJ

        # 3. Get Buffer Accesses
        wbuf_read = wk_layer.get_num_weights()
        wbuf_fill = wk_layer.get_num_weights()
        if not wbuf_dram:
            wbuf_fill = 0
        ifbuf_read = col_fold * T * (self.ydim * row_fold - (self.ydim - row_mod))
        if self.ifbuf_size * 8 * 1024 / (self.bit) >= wk_layer.get_num_inputs():
            if status == 'start':
                ifbuf_fill = wk_layer.get_num_inputs()
            else:
                ifbuf_fill = 0
        else:
            if status == 'start':
                ifbuf_fill = self.ifbuf_size * 1024 + (wk_layer.get_num_inputs() - self.ifbuf_size * 1024) * col_fold
            else:
                ifbuf_fill = (wk_layer.get_num_inputs() - self.ifbuf_size * 1024) * col_fold
        ofbuf_update = row_fold * T * (self.xdim * col_fold - (self.xdim - col_mod))
        if status == 'last':
            ofbuf_fill = wk_layer.get_num_outputs()
        else:
            if self.ofbuf_size * 8 * 1024 / (self.bit) >= wk_layer.get_num_outputs():
                ofbuf_fill = 0
            else:
                ofbuf_fill = wk_layer.get_num_outputs() - self.ofbuf_size * 1024

        

        # 4. Get Buffer Energy
        wbuf_energy = (wbuf_read + wbuf_fill) * self.wbuf_energy / 1e6  # uJ
        ifbuf_energy = (ifbuf_read + ifbuf_fill) * self.ifbuf_energy / 1e6  # uJ
        ofbuf_energy = (ofbuf_update + ofbuf_fill) * self.ofbuf_energy / 1e6  # uJ

        # 5. Get DRAM Energy
        dram_energy = (wbuf_fill + ifbuf_fill + ofbuf_fill) * self.dram_access_energy / 1e6  # uJ

        # 6-a. Get Wire Energy (Systolic Array)
        wire_buf_pe_energy = (ofbuf_update + wbuf_read + ifbuf_read) * self.hb_energy / 1e6
        # wire_buf_pe_energy = (wbuf_read + ifbuf_read + ofbuf_update) * 0.5*abs(np.sqrt(self.pe_total_area)-np.sqrt(self.pe_sram_area)) * self.wire_energy * self.bit / 1e6  # uJ  
        # Because the buffer is right on top of the compute array -> instead cause hybrid bonding overhead
        wire_buf_dram_energy = (wbuf_fill + ifbuf_fill + ofbuf_fill) * self.buf_to_dram_wire * self.wire_energy * self.bit / 1e6  + \
        (wbuf_fill + ifbuf_fill + ofbuf_fill) * self.hb_energy / 1e6  # uJ
        if self.pe_ofbuf_bot == True:
            wire_buf_pe_energy = (ofbuf_update * self.buf_to_pe_wire * self.wire_energy * self.bit \
                                  + (wbuf_read + ifbuf_read) * self.hb_energy ) / 1e6
            wire_buf_dram_energy = (wbuf_fill + ifbuf_fill + ofbuf_fill) * self.buf_to_dram_wire * self.wire_energy * self.bit / 1e6  + \
        (wbuf_fill + ifbuf_fill) * self.hb_energy / 1e6  # uJ
        # 6-b. Get Wire Energy (DCIM)
        wire_buf_pe_energy_dcim = (ofbuf_update + wbuf_read + ifbuf_read) * self.hb_energy / 1e6
        # wire_buf_pe_energy_dcim = (wbuf_read + ifbuf_read + ofbuf_update) * 0.5*abs(np.sqrt(self.dcim_total_area)-np.sqrt(self.dcim_sram_area)) * self.wire_energy * self.bit / 1e6
        wire_buf_dram_energy_dcim = (wbuf_fill + ifbuf_fill + ofbuf_fill) * self.buf_to_dram_wire_dcim * self.wire_energy * self.bit / 1e6 + \
        (wbuf_fill + ifbuf_fill + ofbuf_fill) * self.hb_energy / 1e6  # uJ
        if self.dcim_ofbuf_bot == True:
            wire_buf_pe_energy_dcim = (ofbuf_update * self.buf_to_dcim_wire * self.wire_energy * self.bit \
                                  + (wbuf_read + ifbuf_read) * self.hb_energy ) / 1e6
            wire_buf_dram_energy_dcim = (wbuf_fill + ifbuf_fill + ofbuf_fill) * self.buf_to_dram_wire_dcim * self.wire_energy * self.bit / 1e6  + \
        (wbuf_fill + ifbuf_fill) * self.hb_energy / 1e6  # uJ

        # 7. 3D Wire Energy
        hb_energy = (ofbuf_update + wbuf_read + ifbuf_read + wbuf_fill + ifbuf_fill + ofbuf_fill) * self.hb_energy / 1e6

        # 7-a. Gather Energy (Systolic Array)
        sys_buffer_energy = [wbuf_energy, ifbuf_energy, ofbuf_energy]
        sys_wire_energy = [wire_buf_pe_energy, wire_buf_dram_energy, hb_energy]
        sys_total_energy = sum(sys_buffer_energy) + sum(sys_wire_energy) + dram_energy + total_pe_energy
        # 7-b. Gather Energy (DCIM)
        dcim_buffer_energy = [0, ifbuf_energy, ofbuf_energy]
        dcim_wire_energy = [wire_buf_pe_energy_dcim, wire_buf_dram_energy_dcim, hb_energy]
        dcim_total_energy = sum(dcim_buffer_energy) + sum(dcim_wire_energy) + dram_energy + total_dcim_energy

        # 8. Gather Buffer Accesses
        buf_access = [wbuf_read, ifbuf_read, ofbuf_update, wbuf_fill, ifbuf_fill, ofbuf_fill]

        sys_result = ['3d_systolic', cycles, util, sys_total_energy, total_pe_energy, sys_wire_energy[0], 
                      sys_buffer_energy, sys_wire_energy[1], dram_energy, sys_wire_energy[2],
                      buf_access, self.pe_total_area, self.pe_ofbuf_bot]
        dcim_result = ['3d_dcim', cycles, util, dcim_total_energy, total_dcim_energy, dcim_wire_energy[0],
                       dcim_buffer_energy, dcim_wire_energy[1], dram_energy, sys_wire_energy[2],
                       buf_access, self.dcim_total_area, self.dcim_ofbuf_bot]
        wk_layer.save_sim_results(sys_result)
        wk_layer.save_sim_results(dcim_result)