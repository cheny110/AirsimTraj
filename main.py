from Quadrotor import Quadrotor
from MPCController import TrajectoryMPC
import numpy as np
from rich.console import Console
from rich.progress import track
from rich.traceback import install

SIM_TIME =10
TIME_INTERVAL =0.02
N = 30
MAX_THRUST=4.179446268*4

Wx = np([16,11,13,2.0,2.1,2.3])
Wu = np([17,13,76,0.058])
logger =Console()

if __name__ =="__main__":
    install(show_locals=True)
    quadrotor =Quadrotor(max_thrust=MAX_THRUST,interval=TIME_INTERVAL)
    string=logger.input("Need to generate refercence control?(Y/N):")
    if string.lower() =="y":
        quadrotor.reset()
        quadrotor.recordTrajectory()
        quadrotor.reset()
        logger.print("Reference control generated.",style="blue on white")
    else:
        logger.log("Ignore reference trajectory generation.")
    mpc =TrajectoryMPC(quadrotor,Wx,Wu,TIME_INTERVAL,N)
    mpc.setReferenceControl(mpc.quadrotor.controls)
    mpc.loadReferenceTrajectory()
    hist_thrust =[]
    hist_phi=[]
    hist_theta=[]
    hist_psi=[]
    
    logger.log("start solving...", style="blue")
    for iter in track(range(int(SIM_TIME/TIME_INTERVAL))):
        next_al_trajectory = mpc.desiredState(N,iter)
        next_al_control = mpc.desiredControl(N,iter)
        xnc =mpc.desiredXnc(N)
        res =mpc.solve(next_al_trajectory,next_al_control,xnc)
    