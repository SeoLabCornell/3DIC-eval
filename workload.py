# This is the class to load the workload specification and store the simulation result.
class wk_layer:
    def __init__(self, cur_wk, layer_num):
        self.layer_num = layer_num
        self.name = cur_wk[0]
        self.ifmap_H = int(cur_wk[1])
        self.ifmap_W = int(cur_wk[2])
        self.filter_H = int(cur_wk[3])
        self.filter_W = int(cur_wk[4])
        self.num_channel = int(cur_wk[5])
        self.num_filter = int(cur_wk[6])
        self.stride = int(cur_wk[7])
        self.ofmap_H = int(1 + (self.ifmap_H - self.filter_H) / self.stride)
        self.ofmap_W = int(1 + (self.ifmap_W - self.filter_W) / self.stride)
    

        self.sa_2d_result = []
        self.dcim_2d_result = []
        self.sa_3d_divM_result = []
        self.dcim_3d_divM_result = []
        self.sa_3d_divC_result = []
        self.dcim_3d_divC_result = []
        self.sa_3d_result = []
        self.dcim_3d_result = []
        
    def get_num_weights(self):
        return self.filter_H * self.filter_W * self.num_filter * self.num_channel

    def get_num_inputs(self):
        return self.ifmap_H * self.ifmap_W * self.num_channel

    def get_num_outputs(self):
        return self.ofmap_H * self.ofmap_W * self.num_filter

    def get_num_computes(self):
        return self.filter_H * self.filter_W * self.num_channel * self.num_filter * self.ofmap_H * self.ofmap_W

    def get_layer_name(self):
        return self.name

    def save_sim_results(self, result):
        if result[0] == '2d_systolic':
            self.sa_2d_result.append(result)
            
        elif result[0] == '2d_dcim':
            self.dcim_2d_result.append(result)
            
        elif result[0] == '3d_systolic_combi_div_m':
            self.sa_3d_divM_result.append(result)
            
        elif result[0] == '3d_dcim_combi_div_m':
            self.dcim_3d_divM_result.append(result)
            
        elif result[0] == '3d_systolic_combi_div_c':
            self.sa_3d_divC_result.append(result)
            
        elif result[0] == '3d_dcim_combi_div_c':
            self.dcim_3d_divC_result.append(result)
            
        elif result[0] == '3d_systolic':
            self.sa_3d_result.append(result)
            
        elif result[0] == '3d_dcim':
            self.dcim_3d_result.append(result)

        else:
            print('Saving failed for name: {}'.format(result[0]))