
import sys
import os
from quadrotor.Quadrotor import Quadrotor
from rich.console import Console
from rich.progress import track
from time import sleep
from math import cos
import numpy as np
SAVE_DATA_FILE= "dnn_record_dataset.npy"
SAVE_GT_FILE ="dnn_record_gt.npy"
SAMPLE_NUM = 50000
RECORD_RATE = 50
RESULT_INTERVAL = 0.8

if __name__ == "__main__":
    logger =Console()
    quadrotor =Quadrotor()
    quadrotor.reset()
    datas =[]
    quadrotor.lock_.acquire_lock()
    quadrotor.rotor.enableApiControl(False)
    quadrotor.lock_.release_lock()
    bias = RESULT_INTERVAL*RECORD_RATE
    logger.log("Start recording, please fly the quadrotor manually.",style="blue on white")
    for i in track(range(SAMPLE_NUM+bias)):
        x,y,z = quadrotor.position.x_val,quadrotor.position.y_val,quadrotor.position.z_val
        u,v,w = quadrotor.velocity.x_val,quadrotor.velocity.y_val,quadrotor.velocity.z_val
        ax,ay,az = quadrotor.acceleration.x_val,quadrotor.acceleration.y_val,quadrotor.acceleration.z_val
        dot_phi,dot_theta,dot_psi = quadrotor.angular_velocity.x_val,quadrotor.angular_velocity.y_val,quadrotor.angular_velocity.z_val
        phi,theta,psi = quadrotor.roll,quadrotor.pitch,quadrotor.yaw
        thrust =abs(quadrotor.mass*(az-quadrotor.g)/(cos(phi)*cos(theta)))
        datas.append([x,y,z,u,v,w,ax,ay,az,dot_phi,dot_theta,dot_phi,phi,theta,psi,thrust])
        sleep(1/RECORD_RATE)
    
    
    for i in range(len(datas)-bias):
        datas[i][:6]=datas[i+bias][:6]
        
    logger.log(f"Record dataset finished, data wouble be saved to {SAVE_GT_FILE}",style="blue on white")
    datas=np.array(datas)
    np.save(f"./data/{SAVE_GT_FILE}",datas)
    
    