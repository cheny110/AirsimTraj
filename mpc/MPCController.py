import casadi as ca
import numpy as np
import math
from quadrotor.Quadrotor import Quadrotor
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
        self.next_states=np.zeros((self.N,6),np.float32)
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
        self.xk = self.optimizer.variable(self.N,6)
        self.xk_ref =self.optimizer.parameter(self.N,6)
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
        for k in range(self.N-1):
            x_next = self.fd(self.xk[k,:],self.uk[k,:])
            self.optimizer.subject_to(self.xk[k+1,:]==x_next)
        self.optimizer.subject_to(self.xnc == self.xk[-2,:])
        
        # 目标函数
        cost =0
        for k in range(self.N-1):
            cost+= ca.mtimes( [(self.xk[k,:] -self.xk_ref[k,:]),ca.diag(self.w_x), (self.xk[k,:] -self.xk_ref[k,:]).T])
            cost +=ca.mtimes( [(self.uk[k,:] -self.uk_ref[k,:]),ca.diag(self.w_u), (self.uk[k,:] -self.uk_ref[k,:]).T])
        cost+= ca.mtimes([ (self.xnc-self.xnc_ref), ca.diag(self.w_nc), (self.xnc-self.xnc_ref).T])
        cost/=2
        self.optimizer.minimize(cost)
        
        # 设置上下限
        self.optimizer.subject_to(self.optimizer.bounded(
            [-180*math.pi/180,-180*math.pi/180,-math.pi,0],
            self.uk[:,:].T, 
            [180*math.pi/180,180*math.pi/180,math.pi,self.quadrotor.max_thrust]
        ))
        
        opts_settings={
            'ipopt.max_iter':3000,
            'ipopt.print_level':0,
            'print_time':0,
            'ipopt.acceptable_tol':1e-8,
            'ipopt.acceptable_obj_change_tol':1e-6
        }
        self.optimizer.solver('ipopt',opts_settings)
    
    def fd(self,x,u:ca.MX): # u: roll,pitch,yaw,T
        position = x[:3]
        velocity = x[3:] #body frame velocity
        phi,theta,psi = u[0],u[1],u[2]
        p,q,r = self.quadrotor.angular_velocity #三轴角速度
        reb = self.calculateReb(phi,theta,psi)
        F=reb*ca.vertcat(0,0 ,u[3])
        Fx,Fy,Fz =F[0],F[1],F[2]
        
        velocity_world = ca.mtimes(reb,velocity.T)
        
        dot_velocity = ca.horzcat(
           r*velocity[1] -q*velocity[2] +self.g*theta.sin() + Fx/self.quadrotor.mass,
           p*velocity[2] -r*velocity[0] -self.g*phi.sin()*theta.sin()+Fy/self.quadrotor.mass,
           q*velocity[0] -p*velocity[1] -self.g*phi.cos()*theta.sin() +Fz/self.quadrotor.mass
        )
        position_next =position + velocity_world.T*self.T
        velocity_next = velocity + dot_velocity* self.T

        x_next = ca.horzcat(position_next,velocity_next)
        return x_next
    
    def h(self,x,u):
        return np.concatenate([self.quadrotor.position.to_numpy_array,self.quadrotor.orientation.to_numpy_array],axis=0)
    
    def shift(self,u, x_n):
        '''
            @brief: 类似stack 一样移除最前的元素，在末尾更新加入最新的一个元素
        '''
        u_end = np.concatenate((u[1:], u[-1:]))
        x_n = np.concatenate((x_n[1:], x_n[-1:]))
        return u_end, x_n
    
    
    
    def solve(self,next_trajectory,next_controls,Xnc_ref):
        self.optimizer.set_value(self.xk_ref,next_trajectory.T)
        self.optimizer.set_value(self.uk_ref,next_controls.T)
        self.optimizer.set_value(self.w_x,self.Wx)
        self.optimizer.set_value(self.w_u,self.Wu)
        self.optimizer.set_value(self.w_nc,self.Wnc)
        self.optimizer.set_initial(self.xk,self.next_states)
        self.optimizer.set_initial(self.uk, self.u0)
        self.optimizer.set_value(self.xnc_ref,Xnc_ref)

        sol=self.optimizer.solve()
        u_res =sol.value(self.uk)
        x_m = sol.value(self.xk)
        self.u0 ,self.next_states =self.shift(u_res,x_m)
        return u_res[0],x_m[0]
    
    def loadReferenceTrajectory(self,file="record_rc.npy"):
        self.logger.log(f"Load reference trajectory from local file {file}")
        data =np.load(file,allow_pickle=True).item()
        self.ref_xs = data["x"]
        self.ref_ys = data["y"]
        self.ref_zs= data["z"]
        self.ref_us = data["u"]
        self.ref_vs = data["v"]
        self.ref_ws =data["w"]
        self.setReferenceControl(np.array([data["phi"],data["theta"],data["psi"],data["thrust"]]).T)
        self.ref_trajectory_len = len(self.ref_xs)
    
    def setReferenceControl(self,control):
        self.ref_controls = control
        if(type(control)!=np.ndarray):
            self.ref_controls=np.array(self.ref_controls)
            
    def calculateReb(self,phi,theta,psi):
        l1=ca.horzcat(theta.cos()*psi.cos(), phi.sin()*theta.sin()*psi.cos(), phi.cos()*theta.sin()*psi.cos()+phi.sin()*psi.sin())
        l2=ca.horzcat(theta.cos()*psi.cos(), phi.sin()*theta.sin()*psi.sin(), phi.cos()*theta.sin()*psi.sin()- phi.sin()*psi.cos())
        l3=ca.horzcat(-theta.sin(),phi.sin()*theta.cos(), phi.cos()*theta.cos())
        reb =ca.vertcat(l1,l2,l3)
        return reb
    
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
            
            x_ref=np.concatenate([x_ref,x_ex],axis=None)
            y_ref=np.concatenate([y_ref,y_ex],axis=None)
            z_ref=np.concatenate([z_ref,z_ex],axis=None)
            u_ref=np.concatenate([u_ref,u_ex],axis=None)
            v_ref=np.concatenate([v_ref,v_ex],axis=None)
            w_ref=np.concatenate([w_ref,w_ex],axis=None)
        
        X=np.array([x_ref,y_ref,z_ref,u_ref,v_ref,w_ref])
        return X
    
    def desiredXnc(self,iter):
        idx =iter+ self.N
        if idx >len(self.ref_xs)-1:
            idx =-1
        x_nc = self.ref_xs[idx]
        y_nc = self.ref_ys[idx]
        z_nc =self.ref_zs[idx]
        u_nc =self.ref_us[idx]
        v_nc =self.ref_vs[idx]
        w_nc=self.ref_ws[idx]
        Xnc = np.array([x_nc,y_nc,z_nc,u_nc,v_nc,w_nc])
        return Xnc
    
    def desiredControl(self,N,idx):
        phi_ref =self.ref_controls[idx:(idx+N),0]
        theta_ref =self.ref_controls[idx:(idx+N),1]
        psi_ref =self.ref_controls[idx:(idx+N),2]
        thrust_ref =self.ref_controls[idx:(idx+N),3]
        
        length = len(thrust_ref)
        if length<N:
           phi_ex =np.ones(N-length)*phi_ref[-1]
           theta_ex = np.ones(N-length)*theta_ref[-1]
           psi_ex =np.ones(N-length)*psi_ref[-1]
           thrust_ex =np.ones(N-length)*thrust_ref[-1]
           
           phi_ref=np.concatenate([phi_ref,phi_ex],axis=None)
           theta_ref=np.concatenate([theta_ref,theta_ex],axis=None)
           psi_ref=np.concatenate([psi_ref,psi_ex],axis=None) 
           thrust_ref=np.concatenate([thrust_ref,thrust_ex],axis=None)

        U= np.array([phi_ref,theta_ref,psi_ref,thrust_ref])
        return U