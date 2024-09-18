import matplotlib.backends
from Quadrotor import Quadrotor
from MPCController import TrajectoryMPC
import numpy as np
from rich.console import Console
from rich.progress import track
from rich.traceback import install
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from plotting_3d import Plotting

SIM_TIME =10
TIME_INTERVAL =0.02
N = 30
MAX_THRUST=4.179446268*4

Wx = np.array([16,11,13,2.0,2.1,2.3])
Wu = np.array([17,13,76,0.058])
logger =Console()

if __name__ =="__main__":
    #install(show_locals=True)
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
    hist_time=[]
    hist_x=[]
    hist_y=[]
    hist_z=[]
    logger.log("start solving...", style="blue")
    for iter in track(range(int(SIM_TIME/TIME_INTERVAL))):
        next_al_trajectory = mpc.desiredState(N,iter)
        next_al_control = mpc.desiredControl(N,iter)
        xnc =mpc.desiredXnc(N)
        try:
            u_res,x =mpc.solve(next_al_trajectory,next_al_control,xnc)
            phi,theta,psi,thrust =u_res
            x,y,z =x[:3]
        except Exception as e:
            logger.log("Sovler stop abnormally!!!",style="red on white")
            logger.log(e)
            break
        hist_phi.append(phi)
        hist_theta.append(theta)
        hist_psi.append(psi)
        hist_thrust.append(thrust)
        hist_time.append(iter*TIME_INTERVAL)
        hist_x.append(x)
        hist_y.append(y)
        hist_z.append(z)
    #draw reslut
    plt.figure()
    plt.subplot(311)
    plt.plot(hist_time,hist_thrust)
    plt.xlabel("time: s")
    plt.ylabel("thrust: N")
    
    plt.subplot(312)
    plt.xlabel("time: s")
    plt.ylabel("orientation: rad")
    plt.plot(hist_time,hist_phi)
    plt.plot(hist_time,hist_theta)
    plt.plot(hist_time,hist_psi)
    plt.savefig("result.png")

    
    plot = Plotting("Quadrotor")
    plot.plot_path([hist_x,hist_y,hist_y],"quadrotor")
    reference = [mpc.ref_xs,mpc.ref_ys,mpc.ref_zs]
    plot.plot_path(reference, "reference")
    #save result
    save_data = {
        "phi":hist_phi,
        "theta":hist_theta,
        "psi": hist_psi,
        "thrust":hist_thrust
    }
    np.save("result_controls.npy",save_data,allow_pickle=True)

    