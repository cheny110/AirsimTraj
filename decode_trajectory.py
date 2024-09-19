
from rich.console import Console
from rich.progress import track
import numpy as np
from math import atan2,sqrt,acos,sin,cos,asin
logger= Console()
interval =0.02 #s
mass =1
g=9.8

if __name__=="__main__":
    states = np.load("record_states.npy",allow_pickle=True).item()
    xs= states["x"]
    ys= states["y"]
    zs =states["z"]
    us =states["u"]
    vs= states["v"]
    ws= states["w"]
    
    phis =thetas=psis=thrusts=[]
    phi_pre,theta_pre,psi_pre =0,0,0
    
    T= 1.2*mass*g
    for i in track(range(len(xs)-1)):
        dot_u = (us[i+1] - us[i])/interval
        dot_v = (vs[i+1] - vs[i])/interval
        dot_w = (ws[i+1] - ws[i])/interval
        
        u_ref = (xs[i+1] - xs[i])/interval
        v_ref =(ys[i+1] - ys[i])/interval
        w_ref =(zs[i+1] - ws[i]) /interval
        
        # 计算各点参考角度
        psi = atan2(v_ref,(u_ref+1e-2))
        theta = atan2((dot_u*sin(psi) -dot_v*cos(psi)),(dot_w+g))
        phi =asin(max(min(1,(dot_u*cos(psi)+dot_v*sin(psi))/(T/mass)),-1))
        phis.append(phi)
        thetas.append(theta)
        psis.append(psi)
        
    result = {
        "phi": phis,
        "theta":thetas,
        "psi": psis,
        "thrust":thrusts 
    }
        
    np.save("decode_rc",result)