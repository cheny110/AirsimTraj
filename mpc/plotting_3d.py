import numpy as np
import matplotlib.pyplot as plt
 
class Plotting:
    def __init__(self, name, xlim=[-10,10], ylim=[-10,10], zlim=[-10,50], is_grid=True):
        self.fig = plt.figure()
        self.ax = plt.axes(projection ='3d')
        self.ax.set_title(name)
        self.ax.grid(is_grid)
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_zlim(zlim)
        self.ax.set_xlabel('x [m]')
        self.ax.set_ylabel('y [m]')
        self.ax.set_zlabel('z [m]')
    
    def plot_path(self, path,label_):
        path = np.array(path)
        self.ax.plot(path[:,0], path[:,1], -path[:,2],label=label_)