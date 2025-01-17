import numpy as np
import json

# Following code snippet and corresponding parameters have been imported and modified from ACT (Gupta et al. ISCA'22) and 3D-Carbon (Zhao et al. DAC'24)
# --------------------- Start of snippet ----------------------------
with open("parameters/configs/die_yield.json", 'r') as f:
      yield_config = json.load(f)
with open("parameters/configs/bonding_yield.json", 'r') as f:
      bonding_yield = json.load(f)
with open("parameters/configs/scaling_factors.json", 'r') as f:
      scaling = json.load(f)
with open("parameters/packaging/epa_package.json", 'r') as f:
     Package_EPA = json.load(f)
with open("parameters/packaging/epa_bonding.json", 'r') as f:
     Bonding_EPA = json.load(f)
with open("parameters/packaging/epa_substrate.json", 'r') as f:
     substrate_EPA = json.load(f)
with open("parameters/logic/epa.json", 'r') as f:
     EPA = json.load(f)
with open("parameters/logic/gpa.json", 'r') as f:
     GPA = json.load(f)
with open("parameters/logic/mpa.json", 'r') as f:
     MPA = json.load(f)
with open("parameters/logic/epa_BEOL_perlayer.json", 'r') as f:
     BEPL_EPA = json.load(f)
with open("parameters/configs/die_yield.json", 'r') as f:
     yield_config= json.load(f)
with open("parameters/configs/layer_config.json", 'r') as f:
    layer_config= json.load(f)
with open("parameters/carbon_intensity/manufacturing_location.json", 'r') as f:
    carbon_intensity_config= json.load(f)  

wafer_area=900*np.pi/4

class die:
    def __init__(self,tech:int,name='None',beta=400*1e6,area=0,p1=0.525,feature_size=0,gnumber=0,wafer_diam=300,layer=0,layer_sensitive=1,TSVexist=0,neighborgnumber=0,IO=0):
        
        with open("parameters/configs/layer_config.json", 'r') as f:
            layer_config= json.load(f)
        key=str(tech)+'nm'
        self.areaestimate=0
        self.name=name
        
        if tech not in [3,5,7,8,10,12,14,20,28]:
            raise ValueError("technode(nm) is out of range")
        key=str(tech)+'nm'
        alpha=yield_config[key][1]
        D0=yield_config[key][0]
        self.gnumber=gnumber
        self.layer_sensitive=layer_sensitive
        self.IO=IO
        self.TSVexist=TSVexist
        if not feature_size :
                self.feature_size=tech*1e-9
        # gnumber and area must need one
        if not gnumber and not area:
            raise ValueError("At least one of parameter 'gate number' or 'area' must be provided")
        else:
            
            self.area=area

            self.p=p1
            self.beta=beta
            self.layer=layer
            self.Yield = (1+self.area*D0/alpha)**(-alpha) 
            self.tech = tech
            self.alpha = alpha
            self.waferdiam = wafer_diam
            self.waferarea = 900*np.pi/4
            self.D0 = D0
            self.DPW = int(((np.pi*wafer_diam**2/4/self.area)-np.pi*wafer_diam/(np.sqrt(2*self.area))))
            self.neighborgnumber = neighborgnumber

def die_carbon(dies,number,carbon_intensity,TSV_sensitive=0):
    if not TSV_sensitive:TSV_carbon=0
    else:TSV_carbon=Bonding_EPA["TSV"]*carbon_intensity*dies.area*(number+TSV_sensitive-2)
    key=str(dies.tech)+'nm'
    if dies.layer_sensitive:
        layer_delta=dies.layer-layer_config[key]
        gamma=(EPA[key]+layer_delta*BEPL_EPA[key])/EPA[key]
        diecarbon=((((EPA[key]+layer_delta*BEPL_EPA[key])*carbon_intensity+gamma*GPA[key]+MPA[key])*wafer_area/dies.DPW/dies.Yield))*number+TSV_carbon
    else: diecarbon=((((EPA[key])*carbon_intensity+GPA[key]+MPA[key])*wafer_area/dies.DPW/dies.Yield))*number+TSV_carbon
    return(diecarbon)

def die_carbon_without_yield(dies,number,carbon_intensity,TSV_sensitive=0):
    key=str(dies.tech)+'nm'
    if not TSV_sensitive:TSV_carbon=0
    else:TSV_carbon=Bonding_EPA["TSV"]*carbon_intensity*dies.area*(number+TSV_sensitive-2)

    if dies.layer_sensitive:
        layer_delta=dies.layer-layer_config[key]
        gamma=(EPA[key]+layer_delta*BEPL_EPA[key])/EPA[key]
        diecarbon=(((EPA[key]+layer_delta*BEPL_EPA[key])*carbon_intensity+gamma*GPA[key]+gamma*MPA[key])*wafer_area/dies.DPW)*number+TSV_carbon
    else: diecarbon=(((EPA[key])*carbon_intensity+GPA[key]+MPA[key])*wafer_area/dies.DPW)*number+TSV_carbon
    return diecarbon

class Hybrid_bonding:
    def __init__(self,die_dict:dict,Manufacturing_location="world",packagearea=0,method="D2W",test=0,F2F_F2B="F2B", is_2d=False):
        self.test=test 
        self.diedict=die_dict
        i=0
        keys=list(die_dict.keys())
        if F2F_F2B=="F2F":
            
            for dies in die_dict:
                if i == 0:
                    dies.TSVexist=1
                    try:
                        dies.neighborgnumber=keys[i+1].gnumber
                    except:
                        dies.neighborgnumber=0
                elif i != 1:
                    dies.TSVexist=1
                    dies.neighborgnumber=keys[i-1].gnumber
               

        elif F2F_F2B=="F2B":
            
            for dies in die_dict:
                
                if i!=len(keys)-1:
                    dies.TSVexist=1
                    dies.neighborgnumber=keys[i-1].gnumber
                i+=1
        
        else: raise ValueError("F2F/F2B methods must in F2F or F2B!")
                 
        carbon_intensity=carbon_intensity_config[Manufacturing_location]
        diecarbon=[]
        self.dienumber=0
        self.diefootprint=0
        self.diefootprint=list(die_dict.keys())[0].area
        
        if method=="W2W":
            self.dieyield=1
            for dies in die_dict:
                
                self.dieyield*=dies.Yield**die_dict[dies]
                self.dienumber+=die_dict[dies]
            self.bondingyield=bonding_yield["Hybridbonding_W2W"]**(self.dienumber)
            i=0
            for dies in die_dict:
                if i==0:
                     diedata=die_carbon_without_yield(dies,die_dict[dies],carbon_intensity,TSV_sensitive=1)/self.bondingyield/self.dieyield
                else:diedata=die_carbon_without_yield(dies,die_dict[dies],carbon_intensity,TSV_sensitive=2)/self.bondingyield/self.dieyield
                diecarbon.append(diedata)
                i+=1
                
            self.diecarbon=diecarbon
            self.bondingcarbon=self.diefootprint*carbon_intensity*Bonding_EPA["DBI"]*(self.dienumber-1)/self.bondingyield/self.dieyield
            if not packagearea:
                self.packagearea=self.diefootprint*scaling["package"]
            else:self.packagearea=packagearea
            self.packagecarbon=self.packagearea*Package_EPA['FCBGA']*carbon_intensity

        elif method=="D2W":
            self.dieyield=1
            self.bondingcarbon=0
            for dies in die_dict:
                self.dieyield*=dies.Yield**die_dict[dies]
            self.bondingyield=bonding_yield["Hybridbonding_D2W"]
            i=0
            for dies in die_dict:
                t=dies.layer
                
                if i==0:
                     diedata=die_carbon(dies,die_dict[dies],carbon_intensity,TSV_sensitive=1)/self.bondingyield
                # else all need TSV
                else:
                     dies.layer=int(dies.layer*2/3)
                     diedata=die_carbon(dies,die_dict[dies],carbon_intensity,TSV_sensitive=2)/self.bondingyield
                
                # This equation has been modified from 3D-carbon to consider HB pitch and pillar size. 
                # HB pitch 5.76um, Pillar size 1.97um x 1.97um - Jeloka et al. IEEE CICC 2022.
                self.bondingcarbon+=dies.area*(1.97/5.76)*(1.97/5.76)*Bonding_EPA["DBI"]*carbon_intensity/self.bondingyield/dies.Yield  
                diecarbon.append(diedata)
                i+=1
                dies.layer=t
               
            self.diecarbon=diecarbon
            if not packagearea:
                self.packagearea=self.diefootprint*scaling["package"]
            else:self.packagearea=packagearea
            self.packagecarbon=self.packagearea*Package_EPA['FCBGA']*carbon_intensity
        else: raise ValueError("The bonding method must be D2W or W2W")
        if self.test: self.carbon=(np.sum(self.diecarbon)+self.packagecarbon+self.bondingcarbon)*1000
        
        else: self.carbon=np.sum(self.diecarbon)+self.packagecarbon+self.bondingcarbon
        self.carbonbreak=[np.sum(self.diecarbon),self.bondingcarbon,0,self.packagecarbon]
    def __str__(self) -> str:
        str1=""
        i=0
        
        if self.test:  
            for dies in self.diedict:
                str1=str1+dies.name+": {:.2f} kg, ".format(self.diecarbon[i])
                i+=1  
            return str1+"bonding: {:.1f} kg, ".format(self.bondingcarbon)+"package: {:.1f} kg, ".format(self.packagecarbon)+"overall embodied carbon: {:.1f} kg ".format((np.sum(self.diecarbon)+self.packagecarbon+self.bondingcarbon))
        else:
             for dies in self.diedict:
                str1=str1+dies.name+": {:.2f} kg, ".format(self.diecarbon[i]/1000)
                i+=1  
             return str1+"bonding: {:.2f} kg, ".format(self.bondingcarbon/1000)+"package: {:.2f} kg, ".format(self.packagecarbon/1000)+"overall embodied carbon: {:.2f} kg".format((np.sum(self.diecarbon)+self.packagecarbon+self.bondingcarbon)/1000)

# --------------------- End of snippet ----------------------------

def cal_yield(area, dd=0.0007, alpha=10, num_die=2):
    return ((1 + dd*area/alpha)**(-alpha))**num_die

def cal_carbon_hb(die_area, tech=28, stacking="D2W", F2F_F2B="F2F", packagearea=0, Manufacturing_location="world", scheme_3d=1, is_2d=False):       

    test = 1
    dies = {}    
    area = 0
    
    for i in range(2):
        layer = 6
        if i == 0:
            dname = 'bot'
        else:
            dname = 'top'
            if scheme_3d == 2:
                layer = 4
        die_key = die(tech, name=dname,gnumber=0,area=die_area,
                      feature_size=0, layer=layer, layer_sensitive=1, IO=0)
        
        die_value = 1
        dies.update({die_key:die_value})
        area += die_key.area * die_value
        
    ic = Hybrid_bonding(dies,method=stacking,Manufacturing_location=Manufacturing_location,packagearea=packagearea,test=test,F2F_F2B=F2F_F2B, is_2d=is_2d)
    carbon = ic.carbon
    
    return ic

def get_total_carbon(ic, is_2d=False):
    if is_2d:
        return [ic.diecarbon[1], 0, 0, ic.packagecarbon]
    else:
        return [ic.diecarbon[0], ic.diecarbon[1], ic.bondingcarbon, ic.packagecarbon]