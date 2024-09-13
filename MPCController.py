import casadi as ca
import numpy as np
import math
from Quadrotor import Quadrotor
from math import sin,cos
from rich.console import Console

class TrajectoryMPC:
    def __init__(self,quad:Quadrotor,Wx:np.ndarray, Wu:np.ndarray, T=0.02,N=30,ratio=1.3) -> None:
        self.T =T
        self.N =N
        self.ratio =ratio
        self.g =9.8
        self.quadrotor = quad
        self.logger=Console()
        self.next_states=np.zeros((self.N+1,6),np.float32)
        self.u0 = np.zeros((self.N,4))
        self.Wx =  Wx                    # weight for state parameter
        self.Wu = Wu
        self.Wnc = Wx*ratio
        self.setupController()
        
        self.logger.print("-----------Trajectory NMPC---------------")
        self.logger.print(f"time interval:{T}")
        self.logger.print(f"Control horizons:{self.N}")
        self.logger.log(f"State weight: {self.Wx}")
        self.logger.log(f"Control weight:{self.Wu}")
        self.logger.print(f"Termial State Weight:{self.Wnc}")
        
    def setupController(self):
        self.optimizer =ca.Opti()
        self.xk = self.optimizer.variable(self.N+1,6)
        self.xk_ref =self.optimizer.parameter(self.N+1,6)
        self.uk =self.optimizer.variable(self.N,4)
        self.uk_ref =self.optimizer.parameter(self.N,4)
        self.xnc =self.optimizer.variable(1,6)
        self.xnc_ref =self.optimizer.parameter(1,6)
        self.w_x =self.optimizer.parameter(6)
        self.w_u =self.optimizer.parameter(4)
        self.w_nc =self.optimizer.parameter(6)
        
        # 初始状态
        self.optimizer.subject_to(self.xk[0,:] == self.xk_ref[0,:])
        self.optimizer.subject_to(self.uk[0,:] == self.uk_ref[0,:])
        for k in range(self.N):
            x_next = self.fd(self.xk[k,:],self.uk[k,:])
            self.optimizer.subject_to(self.xk[k+1,:]==x_next)
        
        
        # 目标函数
        cost =0
        for k in range(self.N):
            cost+= ca.mtimes( [(self.xk[k,:] -self.xk_ref[k,:]),ca.diag(self.w_x), (self.xk[k,:] -self.xk_ref[k,:]).T])
            cost +=ca.mtimes( [(self.uk[k,:] -self.uk_ref[k,:]),ca.diag(self.w_u), (self.uk[k,:] -self.uk_ref[k,:]).T])
        cost+= ca.mtimes([ (self.xnc[self.N,:]-self.xnc_ref[self.N,:]).T, ca.diag(self.w_nc), (self.xnc[self.N,:]-self.xnc_ref[self.N,:])])
        cost/=2
        
        self.optimizer.minimize(cost)
        
        # 设置上下限
        self.optimizer.subject_to(self.optimizer.bounded(
            [-35*math.pi/180,-35*math.pi/180,-math.pi,0.5*self.quadrotor.mass*self.g],
            self.uk, 
            [35*math.pi/180,35*math.pi/180,math.pi,1.8*self.quadrotor.mass*self.g]
        ))
        
        opts_settings={
            'ipopt.max_iter':2000,
            'ipopt.print_level':0,
            'print_time':0,
            'ipopt.acceptable_tol':1e-8,
            'ipopt.acceptable_obj_change_tol':1e-6
        }
        self.optimizer.solve('ipopt',opts_settings)
    
    def fd(self,x,u:ca.MX): # u: roll,pitch,yaw,T
        position = x[:3]
        velocity = x[3:]
        phi,theta,psi = u[0],u[1],u[2]
        p,q,r = self.quadrotor.angular_velocity #三轴角速度
        F=u[3] * ca.horzcat(theta.sin(), -phi.sin()*theta.sin(),phi.cos(),theta.sin())
        Fx,Fy,Fz =F[0],F[1],F[2]
        
        dot_velocity = ca.horzcat(
           r*velocity[1] -q*velocity[2] +self.g*theta.sin() + Fx/self.quadrotor.mass,
           p*velocity[2] -r*velocity[0] -self.g*phi.sin()*theta.sin()+Fy/self.quadrotor.mass,
           q*velocity[0] -p*velocity[1] -self.g*phi.cos()*theta.sin() +Fz/self.quadrotor.mass
        )
        position_next =position + velocity*self.T
        velocity_next = velocity + dot_velocity* self.T

        x_next = ca.horzcat(position_next,velocity_next)
        return x_next
    
    def h(self,x,u):
        return np.concatenate([self.quadrotor.position.to_numpy_array,self.quadrotor.orientation.to_numpy_array],axis=0)
    
    def shift(self,u, x_n):
        u_end = np.concatenate((u[1:], u[-1:]))
        x_n = np.concatenate((x_n[1:], x_n[-1:]))
        return u_end, x_n
    
    def solve(self,next_trajectory,next_controls):
        self.optimizer.set_value(self.xk_ref,next_trajectory)
        self.optimizer.set_value(self.uk_ref,next_controls)
        self.optimizer.set_value(self.w_x,self.Wx)
        self.optimizer.set_value(self.w_u,self.Wu)
        self.optimizer.set_value(self.w_nc,self.Wnc)
        self.optimizer.set_initial(self.xk,self.next_states)
        self.optimizer.set_initial(self.uk, self.u0)

        sol=self.optimizer.solve()
        u_res =sol.value(self.uk)
        x_m = sol.value(self.xk)
        self.u0 ,self.next_states =self.shift(u_res,x_m)
        return u_res
    
    def loadReferenceTrajectory(self,file="record_states.npy"):
        self.logger.log(f"Load reference trajectory from local file {file}")
        data =np.load(file,allow_pickle=True).item()
        ref_trajectory = data
        self.ref_xs = data["x"]
        self.ref_ys = data["y"]
        self.ref_zs= data["z"]
        self.ref_us = data["u"]
        self.ref_vs = data["v"]
        self.ref_ws =data["w"]
        self.ref_trajectory_len = len(self.ref_xs)
    
    def setReferenceControl(self,control):
        self.ref_controls = control
        if(type(control)!=np.ndarray):
            self.ref_controls=np.array(self.ref_controls)
    
    def desiredState(self,N,idx):
        x_ref = self.ref_xs[idx:(idx+N)]
        y_ref =self.ref_ys[idx:(idx+N)]
        z_ref =self.ref_zs[idx:(idx+N)]
        u_ref =self.ref_us[idx:(idx+N)]
        v_ref =self.ref_vs[idx:(idx+N)]
        w_ref =self.ref_ws[idx:(idx+N)]
        length =len(x_ref)
        if length< N:
            x_ex = np.ones(N-length)*x_ref[-1]
            y_ex = np.ones(N-length)*y_ref[-1]
            z_ex = np.ones(N-length)*z_ref[-1]
            u_ex =np.ones(N-length)*u_ref[-1]
            v_ex =np.ones(N-length)*v_ref[-1]
            w_ex =np.ones(N-length)*w_ref[-1]
            
            np.concatenate(x_ref,x_ex,axis=None)
            np.concatenate(y_ref,y_ex,axis=None)
            np.concatenate(z_ref,z_ex,axis=None)
            np.concatenate(u_ref,u_ex,axis=None)
            np.concatenate(v_ref,v_ex,axis=None)
            np.concatenate(w_ref,w_ex,axis=None)
        
        X=np.array([x_ref,y_ref,z_ref,u_ref,v_ref,w_ref])
        return X
    
    def desiredControl(self,N,idx):
        phi_ref =self.ref_controls[idx:(idx+N)][0]
        theta_ref =self.ref_controls[idx:(idx+N)][1]
        psi_ref =self.ref_controls[idx:(idx+N)][2]
        thrust_ref =self.ref_controls[idx:(idx+N)][3] *self.quadrotor.max_thrust
        
        length = len(thrust_ref)
        if length<N:
           phi_ex =np.ones(N-length)*phi_ref[-1]
           theta_ex = np.ones(N-length)*theta_ref[-1]
           psi_ex =np.ones(N-length)*psi_ref[-1]
           thrust_ex =np.ones(N-length)*thrust_ex[-1]
           
           np.concatenate([phi_ref,phi_ex],axis=None)
           np.concatenate([theta_ref,theta_ex],axis=None)
           np.concatenate([psi_ref,psi_ex],axis=None) 
           np.concatenate([thrust_ref,thrust_ex],axis=None)

        U= np.array([phi_ref,theta_ref,psi_ref])
        return U