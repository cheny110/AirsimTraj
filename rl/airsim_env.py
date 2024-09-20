
from turtle import done
from gymnasium import Env
from gymnasium import spaces

import numpy as np
from quadrotor.Quadrotor import Quadrotor
from typing import Any, SupportsFloat
from rich.console import Console


class  AirsimEnv(Env):
    HOVER = 0
    A_SETPOINT = 1
    MULTI_SETPOINT = 2
    def __init__(self, initial_state=HOVER,logger:Console=None) -> None:
        super(AirsimEnv,self).__init__()
        self.initial_state = initial_state
        self.state = initial_state
        self.logger = logger if logger else Console()
        self.rotor =Quadrotor()
        self.reward = 0
        wx_low =np.array([0,0,0,0,0,0])
        wx_hight =np.array([1e2,1e2,1e2,1e1,1e1,1e1,1e1])
        wu_low= np.array([0,0,0,-1e-2])
        wu_high =np.array([1e2,1e2,1e2,1e-2])
        action_low = np.concatenate((wx_low,wu_low))
        action_high =np.concatenate(wx_hight,wu_high)
        self.observation_space = spaces.Box(low=np.array([-1e2,-1e2,-1e2,-1e1,-1e1,-1e1]),
                                            high=np.array([1e2,1e2,1e2,1e1,1e1,1e1]),
                                            dtype=np.float32)
        self.action_space = spaces.Box(low=action_low,high=action_high,dtype=np.float32)
        self.logger.log("Airsim RL environment initialized", "blue  on white") 
    
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        self.reward =0  
        self.state =np.array(self.initial_state)
        self.truncated  = False
        self.done =done
        self.info ={}
        self.logger.log("Airsim RL environment reset!", style="blue on white")
    

    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        #TODO: to be implemented 
        #更细状态并计算奖励
        return self.state,self.reward,self.done,self.truncated,self.info
    