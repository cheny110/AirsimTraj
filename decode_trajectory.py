
from rich.console import Console
from rich.progress import track
import numpy as np
from math import atan2,sqrt,acos,sin,cos
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
    for i in track(range(len(xs)-1)):
        dot_u = (us[i+1] - us[i])/interval
        dot_v = (vs[i+1] - vs[i])/interval
        dot_w = (ws[i+1] - ws[i])/interval
        
        u_ref = (xs[i+1] - xs[i])/interval
        v_ref =(ys[i+1] - ys[i])/interval
        w_ref =(zs[i+1] - ws[i]) /interval
        
        # 计算各点参考角度
        dir_vector  =np.array([ xs[i+1] - xs[i],
                                ys[i+1] - ys[i],
                                zs[i+1] - ws[i]]
                                                )
        dir_vector =dir_vector/np.linalg.norm(dir_vector)
        psi = atan2(dir_vector[1],dir_vector[0])
        theta = atan2(dir_vector[2],sqrt(dir_vector[0]**2+dir_vector[1]**2))
        phi = 0
        
        p,q,r =phi-phi_pre, theta-theta_pre, psi-psi_pre
        
        Fx =mass*(dot_u -r*v_ref +q*w_ref -g*sin(theta))
        Fy = mass*(dot_v -p*w_ref+r*u_ref+g*sin(phi)*cos(theta))
        Fz = mass*(q*u_ref-p*v_ref-g*cos(phi)*cos(theta))
        
        F = sqrt(Fx**2+Fy**2+Fz**2)
        
        psis.append(phi)
        thetas.append(theta)
        psis.append(psi)
        thrusts.append(F)
        
        phi_pre,theta_pre,psi_pre =phi,theta,psi
        
        
    result = {
        "phi": phis,
        "theta":thetas,
        "psi": psis,
        "thrust":thrusts 
    }
        
    np.save("decode_rc",result)