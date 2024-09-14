
from Quadrotor import Quadrotor
from rich.console import Console
import numpy as np
from threading import Lock

logger =Console()
lock =Lock()
if __name__ =="__main__":
    rotor =Quadrotor()
    # rotor.reset()
    # logger.print("Now fly reference trajectory,please enable trace line mannually!",style="red")
    # rotor.recordTrajectory()
    rotor.reset()
    logger.print("Now fly result trajectory,please disable trace line")
    trajectory =np.load("result_controls.npy",allow_pickle=True).item()
    phi = trajectory["phi"]
    theta =trajectory["theta"]
    psi = trajectory["psi"]
    thrust= trajectory["thrust"]
    thrust_norm=[]
    rotor.lock_.acquire_lock()
    rotor.takeoff()
    rotor.lock_.release_lock()
    for t in thrust:
        t/=rotor.max_thrust
        thrust_norm.append(t)
    for i,j,k,n in zip(phi,theta,psi,thrust_norm):
        rotor.lock_.acquire_lock()
        rotor.rotor.moveByRollPitchYawThrottleAsync(i,j,k,n,rotor.interval).join()
        rotor.lock_.release_lock()