import numpy as np
import rich
from airsim.client import MultirotorClient,Vector3r
from airsim import to_eularian_angles
from rich.console import Console
from time import sleep
from threading import Thread,Lock
from math import cos,sin
from rich.progress import track

class Quadrotor:
    def __init__(self, pos=[0,0,0], ori=[0,0,0], dpos=[0,0,0], dori=[0,0,0],max_thrust=4.17*4,sim_time=10,interval=0.02):
        # The configuration of quadrotor
        self.pos = np.array(pos)
        self.ori = np.array(ori)
        self.dpos = np.array(dpos)
        self.dori = np.array(dori) # dot orientation
        self.mass = 1.0
        self.sim_time=sim_time
        self.interval =interval 
        self.logger =Console()
        self.lock_=Lock()
        self.rotor = MultirotorClient()
        self.connectAirsim()
        self.max_thrust = max_thrust
        self.controls =self.ref_control()
        self.runBackgroudThreads()
    
    def ref_control(self,sim_time=30,interval=0.02):
        controls=[]
        for i in range(int(sim_time/0.02)):
            phi =1e-4 *i
            theta =1e-4*i
            psi = 0.001*i
            thrust = 1.05*self.mass*9.8/self.max_thrust
            controls.append([phi,theta,psi,thrust])
        return controls
    
    def reset(self):
        self.logger.log("Quadrotor reset!!!",style="red on white")
        self.lock_.acquire_lock()
        self.rotor.reset()
        self.lock_.release_lock()
        self.lock_.acquire_lock()
        self.rotor.enableApiControl(True)
        self.lock_.release_lock()
        self.lock_.acquire_lock()
        self.rotor.armDisarm(True)
        self.lock_.release_lock()
        self.logger.log("Clear all persistent markers.")
        
        #self.rotor.simFlushPersistentMarkers()
    
    def takeoff(self):
        self.rotor.takeoffAsync().join()
    
    def connectAirsim(self):
        self.rotor.confirmConnection()
        self.rotor.enableApiControl(True)
        self.rotor.armDisarm(False)
        
    def runBackgroudThreads(self):
        self.threads_=[]
        update_thread = Thread()
        update_thread.run =self.updateKinematics
        self.threads_.append(update_thread)
        
        for t in self.threads_:
            t.start()
    
    def setTracelineType(self):
        self.lock_.acquire_lock()
        self.rotor.simSetTraceLine([0,0,1,1],1)#blue
        self.lock_.release_lock()
        
    def updateKinematics(self):
        while True:
            self.lock_.acquire_lock()
            self.kinematics=self.rotor.simGetGroundTruthKinematics()
            self.imu =self.rotor.getImuData()
            self.lock_.release_lock()
            self.velocity = self.kinematics.linear_velocity
            self.position =self.kinematics.position
            self.angular_velocity =self.imu.angular_velocity

            self.pitch,self.roll,self.yaw=to_eularian_angles(self.imu.orientation) #机体坐标系
            #self.logger.log(f"roll:{self.roll}, pitch:{self.pitch},yaw:{self.yaw}")
            sleep(0.005)
    
    # def calculateReb(self):
    #     phi,theta,psi = self.roll,self.pitch,self.yaw
    #     self.reb = np.array([
    #         [cos(theta)*cos(psi), sin(phi)*sin(theta)*cos(psi)-sin(psi)*cos(phi), cos(phi)*cos(theta)*cos(psi)+sin(phi)*sin(psi) ],
    #         [cos(theta)*sin(psi), sin(phi)*sin(theta)*sin(psi)+cos(psi)*cos(phi), cos(phi)*sin(theta)*sin(psi)-sin(phi)*cos(psi) ],
    #         [-sin(theta)        , sin(phi)*cos(theta),                            cos(phi)*cos(theta)]
    #     ])
    
    def recordTrajectory(self,save_file="record_states.npy"):
        self.logger.log(f"Start recording trajectory for {self.sim_time} seconds, time interval:{self.interval} s."
                        , style= "white on blue")
        xs=[]
        ys=[]
        zs=[]
        us=[]
        vs=[]
        ws=[]
        for i in track(range(int(self.sim_time/self.interval))):
            x,y,z =self.position.x_val,self.position.y_val,self.position.z_val
            u,v,w =self.velocity.x_val,self.velocity.y_val,self.velocity.z_val
            xs.append(x)
            ys.append(y)
            zs.append(z)
            us.append(u)
            vs.append(v)
            ws.append(w)
            self.lock_.acquire_lock()
            self.rotor.moveByRollPitchYawThrottleAsync(*self.controls[i],duration=self.interval).join()
            self.lock_.release_lock()
        save_data = {
            "x":xs,
            "y":ys,
            "z":zs,
            "u":us,
            "v":vs,
            "w":ws
        }
        self.logger.log(f"Save records to {save_file}")
        np.save(save_file,save_data,allow_pickle=True)
        
        self.logger.log("print saved trajectory in red line for reference",style="white on blue")
        # plot_trajectory =[]
        # for i in range(len(records)):
        #     traj=Vector3r(records[i][0],records[i][1],records[i][2])
        #     plot_trajectory.append(traj)
        # self.lock_.acquire_lock()
        # self.rotor.simPlotPoints(plot_trajectory,size=1,is_persistent=True)
        # self.lock_.release_lock()
    
    
        
    def hover(self):
        self.lock_.acquire_lock()
        self.rotor.hoverAsync()
        self.lock_.release_lock()

if __name__ =="__main__":
    rotor=Quadrotor()
    rotor.reset()
    rotor.setTracelineType()
    rotor.recordTrajectory()
    rotor.hover()
