# import numpy as np
# import matplotlib.pyplot as plt
# import random
# import matplotlib as mpl
# import pandas as pd
# from matplotlib.colors import LinearSegmentedColormap
# from scipy import ndimage
# from matplotlib import colors as mcolors
# import matplotlib.animation as animation   # only needed if you want to make an animation
#
#
#
# data = pd.read_csv('simulation_stats.csv')
# mpl.rcParams['font.family'] = 'Times New Roman'
# mpl.rcParams['font.serif'] = ['Times New Roman']
#
# if not {'Misspelling', 'Spelling', 'Time'}.issubset(data.columns):
#     raise ValueError("CSV 文件中缺少必要的列：Misspelling, Spelling, Time")
#
# x = data['Time']
# y1 = data['Misspelling']
# y2 = data['Spelling']
# class Environment:
#     def __init__(self, N=200, M=200, pp=0.015):
#
#         self.N = N
#         self.M = M
#         self.data_map = np.zeros(shape=(N,M))
#         self.trail_map = np.zeros(shape=(N,M))
#         self.population = int((self.N*self.M)*(pp))
#         self.particles = []
#
#     def read_grid_states(self, filename="./grid_states.txt"):
#         grid_data = []
#         with open(filename, 'r') as file:
#             next(file)
#             for line in file:
#                 parts = line.strip().split(', ')
#                 X = int(parts[0])
#                 Y = int(parts[1])
#                 State = float(parts[2])
#                 grid_data.append((X, Y, State))
#         return grid_data
#
#
#     def populate(self, SA=np.pi/4, RA=np.pi/8, SO=1):
#         grid_data = self.read_grid_states("grid_states.txt")
#
#         for (X, Y, State) in grid_data:
#                 if 0 <= X < self.N and 0 <= Y < self.M:
#                     self.data_map[X, Y] = 1
#                     p = Particle((X, Y), SA, RA, SO)
#                     self.particles.append(p)
#
#
#     def deposit_food(self, pos, strength=3, rad=6):
#         n, m = pos
#         y, x = np.ogrid[-n:self.N-n, -m:self.M-m]
#         mask = x**2 + y**2 <= rad**2
#         self.trail_map[mask] = strength
#
#     def diffusion_operator(self, const=0.8, sigma=2):
#         # print(f"{self.trail_map}")
#         self.trail_map = const * ndimage.gaussian_filter(self.trail_map,sigma)
#
#     def check_surroundings(self, point, angle):
#         n,m = point
#         x = np.cos(angle)
#         y = np.sin(angle)
#
#         if (self.data_map[(n-round(x))%self.N,(m+round(y))%self.M]==0):
#             return ((n-round(x))%self.N,(m+round(y))%self.M)
#         elif (self.data_map[(n-round(x))%self.N,(m+round(y))%self.M]==1):
#             return point
#
#     def motor_stage(self):
#         rand_order = random.sample(self.particles, len(self.particles))
#         for i in range(len(rand_order)):
#             old_x, old_y = rand_order[i].pos
#             new_x, new_y = self.check_surroundings(rand_order[i].pos, rand_order[i].phi)
#             if ((new_x,new_y) == (old_x,old_y)):
#                 rand_order[i].phi = 2*np.pi*np.random.random()
#                 rand_order[i].update_sensors()
#             else:
#                 rand_order[i].pos = (new_x,new_y)
#                 rand_order[i].update_sensors()
#                 self.data_map[old_x,old_y] = 0
#                 self.data_map[new_x,new_y] = 1
#                 rand_order[i].deposit_phermone_trail(self.trail_map)
#
#     def sensory_stage(self):
#         rand_order = random.sample(self.particles, len(self.particles))
#         for i in range(len(rand_order)):
#             rand_order[i].sense(self.trail_map)
#
# class Particle:
#     def __init__(self, pos, SA=np.pi/8, RA=np.pi/4, SO=1):
#         # Initializes physical characteristics of the particle
#         # pos = (n,m) in data map
#         # phi = initial random angle of orientation of the particle [0,2pi]
#         # SA = +- sensor angle wrt phi
#         # SO = sensor offset from body
#         # SS = step size - DONT USE
#         # RA = rotation angle of particle
#         self.pos = pos
#         self.initial_pos = pos
#
#         self.phi = 2*np.pi*np.random.random()
#         self.SA = SA
#         self.RA = RA
#         self.SO = SO
#
#         self.phi_L = self.phi - SA
#         self.phi_C = self.phi
#         self.phi_R = self.phi + SA
#
#     def deposit_phermone_trail(self, arr, strength=1.):
#         n, m = self.pos
#         arr[n,m] = strength
#
#     def update_sensors(self):
#         self.phi_L = self.phi - self.SA
#         self.phi_C = self.phi
#         self.phi_R = self.phi + self.SA
#
#     def get_sensor_values(self, arr):
#         n,m = self.pos
#         row,col = arr.shape
#
#         xL = round(self.SO*np.cos(self.phi_L))
#         yL = round(self.SO*np.sin(self.phi_L))
#         xC = round(self.SO*np.cos(self.phi_C))
#         yC = round(self.SO*np.sin(self.phi_C))
#         xR = round(self.SO*np.cos(self.phi_R))
#         yR = round(self.SO*np.sin(self.phi_R))
#
#         # implement periodic BCs
#         valL = arr[(n-xL)%row,(m+yL)%col]
#         valC = arr[(n-xC)%row,(m+yC)%col]
#         valR = arr[(n-xR)%row,(m+yR)%col]
#
#         return (valL,valC,valR)
#
#     def sense(self, arr):
#         L,C,R = self.get_sensor_values(arr)
#
#         if ((C>L) and (C>R)):
#             self.phi += 0
#             self.update_sensors()
#         elif ((L==R) and C<L):
#             rn = np.random.randint(2)
#             if rn == 0:
#                 self.phi += self.RA
#                 self.update_sensors()
#             else:
#                 self.phi -= self.RA
#                 self.update_sensors()
#         elif (R>L):
#             self.phi += self.RA
#             self.update_sensors()
#         elif (L>R):
#             self.phi -= self.RA
#             self.update_sensors()
#         else:   # all three are the same - stay facing same direction
#             self.phi += 0
#             self.update_sensors()
#
# def scheduler(N=200, M=200, pp=0.37, sigma=0.65, const=0.90,
#               SO=8, SA=np.pi/8, RA=np.pi/4, steps=500,
#               intervals=8, plot=True, animate=False):
#     # The environment is generated with dimensions NxM, where pp% of the environment is populated with particles.
#     # Particles have three key properties:
#     # - Sensor Offset (SO): Determines the movement distance; the higher the value, the faster the decay rate.
#     # - Sensor Angle (SA): Controls the angle range of the left and right sensors, affecting the breadth of perception.
#     # - Rotation Angle (RA): Determines the rotational angle of the particles, influencing the adjustment in their orientation.
#     # The simulation also includes a chemoattractant with two parameters:
#     # - Constant multiplier: Used to scale the influence of the chemoattractant.
#     # - Sigma: A parameter for the Gaussian filter that controls the spread of the attractant.
#     # The simulation evolves over 500 steps, with the option to grab plots at specific intervals.
#     # Users can choose to either plot these intervals or animate the simulation.
#
#     Env = Environment(N, M, pp)
#     Env.populate(SA, RA, SO)
#     if (plot==True):
#         dt = int(steps/intervals)
#         samples = np.linspace(0, dt*intervals, intervals+1)   # integer samples
#         for i in range(steps):
#             Env.diffusion_operator(const,sigma)
#             Env.motor_stage()
#             Env.sensory_stage()
#             if i in samples:
#                 fig = plt.figure(figsize=(8,8),dpi=200);
#                 ax1 = fig.add_subplot(111);
#                 # ax1.imshow(Env.trail_map);
#                 # display some information about parameters used
#                 ax1.text(0,-10,'SA: {:.2f}  SO: {}  RA: {:.2f}  pop: {:.0f}%'.format(np.degrees(SA),SO,np.degrees(RA),pp*100));   # hard code -10, since most likely using big grid
#                 # plt.savefig('sim_t{}.png'.format(i));
#                 plt.clf();
#
#             # 在特定条件下生成图像
#             if i == 299:
#
#                 fig, axs = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'wspace': 0.12})
#                 colors = ['#042C4D', '#255278', '#36658D', '#3E6F98', '#4678A2', '#658FB2', '#84A5C1', '#C2D2E0',
#                           '#FFFFFF']
#                 cmap = LinearSegmentedColormap.from_list('red_yellow', colors[::-1])
#
#                 ax1 = axs[1]
#                 ax1.imshow(Env.trail_map, cmap='Blues')
#
#                 initial_positions = [particle.initial_pos for particle in Env.particles]
#                 initial_x, initial_y = zip(*initial_positions)
#                 ax1.invert_yaxis()  # 颠倒 y 轴方向
#
#                 common_ticks = np.linspace(0, 200, 9)  # 刻度范围从 0 到 200，分成 5 个刻度
#
#                 ax1.set_xlim(0, 200)
#                 ax1.set_ylim(0, 200)
#                 ax1.set_xticks(common_ticks)
#                 ax1.set_yticks(common_ticks)
#                 ax1.set_xticklabels([f"{int(tick)}" for tick in common_ticks], fontsize=14)
#                 ax1.set_yticklabels([f"{int(tick)}" for tick in common_ticks], fontsize=14)
#                 ax1.scatter(initial_y, initial_x, color='grey', s=0.8, label="Initial Positions")
#                 ax1.legend(markerscale=5, loc='upper right',fontsize=12)
#
#                 ax2 = axs[0]
#                 ax2.plot(x, y1, label="Misspelling", color='#C26B61')
#                 ax2.plot(x, y2, label="Spelling", color='#496C88')
#                 ax2.set_xlabel("Time",fontsize=17,fontweight='bold')
#                 ax2.set_ylabel("Count",fontsize=17,fontweight='bold',labelpad=10)
#                 ax2.legend(fontsize=12)
#                 ax2.grid(False)
#
#                 norm = mcolors.Normalize(vmin=0, vmax=1)
#                 sm = plt.cm.ScalarMappable(cmap='Blues', norm=norm)
#                 sm.set_array([])
#                 cbar = plt.colorbar(sm, ax=ax1, shrink=9, aspect=18.5, pad=0.02, fraction=0.05, extend='both')
#                 cbar.set_label('Spelling error propagation intensity', fontsize=16, fontweight='bold')  # 设置颜色条标签
#
#                 ticks = np.linspace(0, 1, num=5)
#                 cbar.set_ticks(ticks)
#                 cbar.set_ticklabels([f"{tick:.2f}" for tick in ticks])
#
#                 ax2.text(-0.18, 1.09, "A", transform=ax2.transAxes, fontsize=24, fontweight='bold', va='top', ha='left')
#                 ax1.text(-0.15, 1.09, "B", transform=ax1.transAxes, fontsize=24, fontweight='bold', va='top', ha='left')
#
#                 axs[0].text(0.5, -0.1, "N", fontsize=17, ha='center', va='center', transform=axs[1].transAxes,
#                             fontweight='bold')
#                 axs[0].text(-0.12, 0.5, "N", fontsize=17, ha='center', va='center', transform=axs[1].transAxes,
#                             fontweight='bold')
#                 axs[0].tick_params(axis='both', which='major', labelsize=14)
#                 axs[1].tick_params(axis='both', which='major', labelsize=14)
#                 axs[0].set_xlim(left=0)  # x轴从0开始
#                 axs[0].set_ylim(bottom=0)  # y轴从0开始
#                 ax2.set_ylim(0, 40000)  # y轴从0到40000
#
#                 plt.tight_layout()
#                 plt.savefig("./simulation.svg")
#                 plt.show()
#
#
#     elif (animate==True):
#         ims = []
#         fig = plt.figure(figsize=(8,8),dpi=100);
#         ax = fig.add_subplot(111);
#         for i in range(steps):
#             Env.diffusion_operator(const,sigma)
#             Env.motor_stage()
#             Env.sensory_stage()
#             txt = plt.text(0,-10,'iteration: {}    SA: {:.2f}    SO: {}    RA: {:.2f}    %pop: {}%'.format(i,np.degrees(SA),SO,np.degrees(RA),pp*100));
#             im = plt.imshow(Env.trail_map, animated=True,cmap='Blues');
#             ims.append([im,txt])
#             initial_positions = [particle.initial_pos for particle in Env.particles]
#             initial_x, initial_y = zip(*initial_positions)
#             ax.scatter(initial_y, initial_x, color='grey', s=0.8, label="Initial Positions")
#         fig.suptitle('Chemoattractant Map');
#         ani = animation.ArtistAnimation(fig,ims,interval=50,blit=True,repeat_delay=1000);
#         ani.save('sim_01210.gif');
#
#
# def main():
#     scheduler(steps=300)
#     return 0
#
# if __name__ == "__main__":
#     main()
#
# -*- coding: utf-8 -*-
"""
Physarum simulation

Based on the work of Jeff Jones, UWE https://uwe-repository.worktribe.com/output/980579
and Sage Jensen https://sagejenson.com/physarum

@author: Amitav Mitra
"""

# handle imports
import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib as mpl
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy import ndimage
from matplotlib import colors as mcolors
import matplotlib.animation as animation  # only needed if you want to make an animation
from matplotlib.lines import Line2D  # 确保这一行存在

# 读取 simulation_stats.csv 文件
data = pd.read_csv('simulation_stats.csv')
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.serif'] = ['Times New Roman']  # 确保 serif 字体为 Times New Roman

# 确保需要的列存在
if not {'Misspelling', 'Spelling', 'Time'}.issubset(data.columns):
    raise ValueError("CSV 文件中缺少必要的列：Misspelling, Spelling, Time")

# 提取数据
x = data['Time']
y1 = data['Misspelling']
y2 = data['Spelling']


class Environment:
    def __init__(self, N=200, M=200, pp=0.015):
        '''
        pp = percentage of the map size to generate population. default 15% - 6000 particles in 200x200 environment
        '''
        self.N = N
        self.M = M
        self.data_map = np.zeros(shape=(N, M))
        self.trail_map = np.zeros(shape=(N, M))
        self.population = int((self.N * self.M) * (pp))
        self.particles = []  # holds particles

    # 修改代码
    def read_grid_states(self, filename="./grid_states.txt"):
        '''读取 grid_states.txt 文件，并返回包含 (X, Y, State) 数据的列表'''
        grid_data = []
        with open(filename, 'r') as file:
            next(file)  # 跳过第一行
            for line in file:
                parts = line.strip().split(', ')
                X = int(parts[0])  # 获取 X 坐标
                Y = int(parts[1])  # 获取 Y 坐标
                State = float(parts[2])  # 获取 State 值
                grid_data.append((X, Y, State))
        return grid_data

    # 修改代码

    def populate(self, SA=np.pi / 4, RA=np.pi / 8, SO=1):
        '''
        randomly populates pp% of the map with particles of:
        SA = Sensor Angle
        RA = Rotation Angle
        SO = Sensor Offset
        '''
        # 源代码
        # while (np.sum(self.data_map) < self.population):   # loop until population size met
        #     rN = np.random.randint(self.N)
        #     rM = np.random.randint(self.M)
        #
        #     if (self.data_map[rN,rM] == 0):
        #         print("1")
        #         p = Particle((rN,rM),SA,RA,SO)
        #         self.particles.append(p)   # list holds particle and its position
        #         self.data_map[rN,rM] = 1   # assign a value of 1 to the particle location
        #     else:
        #         pass
        # 源代码
        # 修改代码
        # Calculate the central region of the map (10 points in both directions)
        # center_row_start = self.N // 2 - 5  # Start of the center 10 points in rows
        # center_row_end = self.N // 2 + 5  # End of the center 10 points in rows
        # center_col_start = self.M // 2 - 5  # Start of the center 10 points in columns
        # center_col_end = self.M // 2 + 5  # End of the center 10 points in columns
        #
        # # Loop through the center 10x10 region and populate particles
        # for i in range(center_row_start, center_row_end):
        #     for j in range(center_col_start, center_col_end):
        #         if self.data_map[i, j] == 0:  # Check if the spot is empty
        #             p = Particle((i, j), SA, RA, SO)
        #             self.particles.append(p)  # Add particle to the list
        #             self.data_map[i, j] = 1  # Mark this location as occupied
        grid_data = self.read_grid_states("grid_states.txt")

        # 遍历每个 (X, Y, State) 数据，如果 State 为 0.0，则将该点标记为 1
        for (X, Y, State) in grid_data:
            if State == 0.0:
                # 检查位置是否在 map 范围内
                if 0 <= X < self.N and 0 <= Y < self.M:
                    self.data_map[X, Y] = 1  # 将该点标记为 1
                    p = Particle((X, Y), SA, RA, SO)  # 创建粒子
                    self.particles.append(p)  # 将粒子添加到列表中
        # 修改代码

    def deposit_food(self, pos, strength=3, rad=6):
        '''
        applies a circular distribution of food to the trail map
        '''
        n, m = pos  # location of food
        y, x = np.ogrid[-n:self.N - n, -m:self.M - m]  # work directly on pixels of the trail map
        mask = x ** 2 + y ** 2 <= rad ** 2  # create circular mask of desired radius
        self.trail_map[mask] = strength

    def diffusion_operator(self, const=0.8, sigma=2):
        '''
        applies a Gaussian filter to the entire trail map, spreading out chemoattractant
        const multiplier controls decay rate (lower = greater rate of decay, keep <1)
        Credit to: https://github.com/ecbaum/physarum/blob/8280cd131b68ed8dff2f0af58ca5685989b8cce7/species.py#L52
        '''
        # print(f"{self.trail_map}")
        self.trail_map = const * ndimage.gaussian_filter(self.trail_map, sigma)

    def check_surroundings(self, point, angle):
        '''
        Helper function for motor_stage()
        Determines if the adjacent spot in the data map is available, based on particle angle
        '''
        n, m = point
        x = np.cos(angle)
        y = np.sin(angle)
        # periodic BCs -> %
        if (self.data_map[(n - round(x)) % self.N, (m + round(y)) % self.M] == 0):  # position unoccupied, move there
            return ((n - round(x)) % self.N, (m + round(y)) % self.M)
        elif (self.data_map[(n - round(x)) % self.N, (m + round(y)) % self.M] == 1):  # position occupied, stay
            return point

    def motor_stage(self):
        '''
        Scheduler function - causes every particle in population to undergo motor stage
        Particles randomly sampled to avoid long-term bias from sequential ordering
        '''
        rand_order = random.sample(self.particles, len(self.particles))
        for i in range(len(rand_order)):
            old_x, old_y = rand_order[i].pos
            new_x, new_y = self.check_surroundings(rand_order[i].pos, rand_order[i].phi)
            if ((new_x, new_y) == (old_x, old_y)):  # move invalid, stay and choose new orientation, update sensors
                rand_order[i].phi = 2 * np.pi * np.random.random()
                rand_order[i].update_sensors()
            else:  # move valid: move there, change value in data map accordingly, deposit trail, AND change particle position
                rand_order[i].pos = (new_x, new_y)
                rand_order[i].update_sensors()
                self.data_map[old_x, old_y] = 0
                self.data_map[new_x, new_y] = 1
                rand_order[i].deposit_phermone_trail(self.trail_map)

    def sensory_stage(self):
        '''
        Makes every particle undergo sensory stage in random order
        '''
        rand_order = random.sample(self.particles, len(self.particles))
        for i in range(len(rand_order)):
            rand_order[i].sense(self.trail_map)


class Particle:
    def __init__(self, pos, SA=np.pi / 8, RA=np.pi / 4, SO=1):
        '''
        Initializes physical characteristics of the particle
        pos = (n,m) in data map
        phi = initial random angle of orientation of the particle [0,2pi]
        SA = +- sensor angle wrt phi
        SO = sensor offset from body
        SS = step size - DONT USE
        RA = rotation angle of particle
        '''
        self.pos = pos
        # 修改代码
        self.initial_pos = pos
        # 修改代码
        self.phi = 2 * np.pi * np.random.random()
        self.SA = SA
        self.RA = RA
        self.SO = SO
        # initialize sensor angles wrt body - will be updated as particle moves
        self.phi_L = self.phi - SA  # left sensor
        self.phi_C = self.phi  # center sensor  - probably redundant, can just use self.phi
        self.phi_R = self.phi + SA  # right sensor

    def deposit_phermone_trail(self, arr, strength=1.):
        '''
        Applies a single trail of chemoattractant at current position
        '''
        n, m = self.pos
        arr[n, m] = strength

    def update_sensors(self):
        '''
        Updates the sensor positions relative to the particle's orientation
        (Left, Center, Right)
        '''
        self.phi_L = self.phi - self.SA
        self.phi_C = self.phi
        self.phi_R = self.phi + self.SA

    def get_sensor_values(self, arr):
        '''
        Finds the value of the chemoattractant at each of the 3 sensors
        Pass the TrailMap array as an argument
        '''
        n, m = self.pos
        row, col = arr.shape

        xL = round(self.SO * np.cos(self.phi_L))
        yL = round(self.SO * np.sin(self.phi_L))
        xC = round(self.SO * np.cos(self.phi_C))
        yC = round(self.SO * np.sin(self.phi_C))
        xR = round(self.SO * np.cos(self.phi_R))
        yR = round(self.SO * np.sin(self.phi_R))

        # implement periodic BCs
        valL = arr[(n - xL) % row, (m + yL) % col]
        valC = arr[(n - xC) % row, (m + yC) % col]
        valR = arr[(n - xR) % row, (m + yR) % col]

        return (valL, valC, valR)

    def sense(self, arr):
        '''
        The particle reads from the trail map, rotates based on chemoattractant
        arr = trail map array
        '''
        L, C, R = self.get_sensor_values(arr)

        if ((C > L) and (C > R)):  # Center > both: stay facing same direction, do nothing
            self.phi += 0
            self.update_sensors()
        elif ((L == R) and C < L):  # L, R are equal, center is less - randomly rotate L/R
            rn = np.random.randint(2)
            if rn == 0:
                self.phi += self.RA
                self.update_sensors()
            else:
                self.phi -= self.RA
                self.update_sensors()
        elif (R > L):
            self.phi += self.RA
            self.update_sensors()
        elif (L > R):
            self.phi -= self.RA
            self.update_sensors()
        else:  # all three are the same - stay facing same direction
            self.phi += 0
            self.update_sensors()


def scheduler(N=200, M=200, pp=0.37, sigma=0.65, const=0.90,
              SO=8, SA=np.pi / 8, RA=np.pi / 4, steps=500,
              intervals=8, plot=True, animate=False):
    # SO代表移动距离 pp总粒子数 const衰减强度，数值越高衰减越快 SA：决定粒子左右传感器的角度范围，影响感知的广度。 RA：决定粒子旋转的角度，影响粒子朝向的调整幅度。
    '''
    generates the environment (NxM) with pp% of environment populated
    particles: Sensor Offset, Sensor Angle, Rotation Angle
    chemoattractant: constant multiplier, sigma (gaussian filter)
    evolve simulation for 500 steps, grab plots at specific intervals
    choice to plot intervals OR animate the desired simulation
    '''
    Env = Environment(N, M, pp)
    Env.populate(SA, RA, SO)
    if (plot == True):
        dt = int(steps / intervals)
        samples = np.linspace(0, dt * intervals, intervals + 1)  # integer samples
        for i in range(steps):
            Env.diffusion_operator(const, sigma)
            Env.motor_stage()
            Env.sensory_stage()
            if i in samples:
                fig = plt.figure(figsize=(8, 8), dpi=200);
                ax1 = fig.add_subplot(111);
                # ax1.imshow(Env.trail_map);
                # display some information about parameters used
                ax1.text(0, -10,
                         'SA: {:.2f}  SO: {}  RA: {:.2f}  pop: {:.0f}%'.format(np.degrees(SA), SO, np.degrees(RA),
                                                                               pp * 100));  # hard code -10, since most likely using big grid
                # plt.savefig('sim_t{}.png'.format(i));
                plt.clf();
            # 修改代码
            # 在最后一帧绘制初始位置
            # if i == 40:
            #     fig = plt.figure(figsize=(8, 8), dpi=200)
            #     ax1 = fig.add_subplot(111)
            #     colors = ['#042C4D', '#255278','#36658D','#3E6F98','#4678A2','#658FB2','#84A5C1','#C2D2E0','#FFFFFF']
            #     cmap = LinearSegmentedColormap.from_list('red_yellow', colors[::-1])
            #
            #     ax1.imshow(Env.trail_map,cmap='Blues')
            #
            #     # 标出所有粒子的初始位置
            #     initial_positions = [particle.initial_pos for particle in Env.particles]
            #     initial_x, initial_y = zip(*initial_positions)
            #     ax1.scatter(initial_y, initial_x, color='grey', s=0.1, label="Initial Positions")  # 用红色标出初始位置
            #     # ax1.set_title('Final Step with Initial Particle Positions')
            #     # ax1.legend()
            #     plt.show()
            #     plt.savefig("./1114.png")
            #     print("hello")
            # 在特定条件下生成图像
            if i == 299:
                # 创建一个包含两个子图的网格
                fig, axs = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'wspace': 0.12})

                # 配置 colormap
                colors = ['#042C4D', '#255278', '#36658D', '#3E6F98', '#4678A2', '#658FB2', '#84A5C1', '#C2D2E0',
                          '#FFFFFF']
                cmap = LinearSegmentedColormap.from_list('red_yellow', colors[::-1])

                # 子图 b: Trail Map with initial particle positions
                ax1 = axs[1]
                ax1.imshow(Env.trail_map, cmap='Blues')

                # 标出所有粒子的初始位置
                initial_positions = [particle.initial_pos for particle in Env.particles]
                initial_x, initial_y = zip(*initial_positions)
                ax1.invert_yaxis()  # 颠倒 y 轴方向
                # 设置子图范围一致
                common_ticks = np.linspace(0, 200, 9)  # 刻度范围从 0 到 200，分成 5 个刻度

                # 应用于 ax1
                ax1.set_xlim(0, 200)
                ax1.set_ylim(0, 200)
                ax1.set_xticks(common_ticks)
                ax1.set_yticks(common_ticks)
                ax1.set_xticklabels([f"{int(tick)}" for tick in common_ticks], fontsize=14)
                ax1.set_yticklabels([f"{int(tick)}" for tick in common_ticks], fontsize=14)
                ax1.scatter(initial_y, initial_x, color='grey', s=0.8, label="Initial Positions")
                ax1.legend(markerscale=5, loc='upper right', fontsize=12)

                # 子图 a: 从 CSV 文件读取数据绘制图
                ax2 = axs[0]
                ax2.plot(x, y1, label="Misspelling", color='#C26B61')
                ax2.plot(x, y2, label="Spelling", color='#496C88')
                ax2.set_xlabel("Time", fontsize=17, fontweight='bold')
                ax2.set_ylabel("Count", fontsize=17, fontweight='bold', labelpad=10)
                ax2.legend(fontsize=12)
                ax2.grid(False)

                norm = mcolors.Normalize(vmin=0, vmax=1)  # 归一化
                sm = plt.cm.ScalarMappable(cmap='Blues', norm=norm)  # 使用渐变色图
                sm.set_array([])  # 设置数组为空
                cbar = plt.colorbar(sm, ax=ax1, shrink=9, aspect=18.5, pad=0.02, fraction=0.05, extend='both')
                cbar.set_label('Spelling error propagation intensity', fontsize=16, fontweight='bold')  # 设置颜色条标签

                # 标记颜色条的刻度
                ticks = np.linspace(0, 1, num=5)
                cbar.set_ticks(ticks)
                cbar.set_ticklabels([f"{tick:.2f}" for tick in ticks])

                ax2.text(-0.18, 1.09, "A", transform=ax2.transAxes, fontsize=24, fontweight='bold', va='top', ha='left')
                ax1.text(-0.15, 1.09, "B", transform=ax1.transAxes, fontsize=24, fontweight='bold', va='top', ha='left')

                axs[0].text(0.5, -0.1, "N", fontsize=17, ha='center', va='center', transform=axs[1].transAxes,
                            fontweight='bold')
                axs[0].text(-0.12, 0.5, "N", fontsize=17, ha='center', va='center', transform=axs[1].transAxes,
                            fontweight='bold')
                axs[0].tick_params(axis='both', which='major', labelsize=14)
                axs[1].tick_params(axis='both', which='major', labelsize=14)
                axs[0].set_xlim(left=0)  # x轴从0开始
                axs[0].set_ylim(bottom=0)  # y轴从0开始
                ax2.set_ylim(0, 40000)  # y轴从0到40000

                # 调整布局
                plt.tight_layout()

                # 保存并展示图像
                plt.savefig("./0121_1.svg")
                plt.show()
                # print("hello")

            # 修改代码

    elif (animate == True):
        # this can take a while for large environments, high population
        # also generates very large .gif files, play with values to get smaller files
        ims = []
        fig = plt.figure(figsize=(8, 8), dpi=100);
        ax = fig.add_subplot(111);
        for i in range(steps):
            Env.diffusion_operator(const, sigma)
            Env.motor_stage()
            Env.sensory_stage()
            txt = plt.text(0, -10,
                           'iteration: {}    SA: {:.2f}    SO: {}    RA: {:.2f}    %pop: {}%'.format(i, np.degrees(SA),
                                                                                                     SO, np.degrees(RA),
                                                                                                     pp * 100));
            im = plt.imshow(Env.trail_map, animated=True, cmap='Blues');
            ims.append([im, txt])
            initial_positions = [particle.initial_pos for particle in Env.particles]
            initial_x, initial_y = zip(*initial_positions)
            ax.scatter(initial_y, initial_x, color='grey', s=0.8, label="Initial Positions")
        fig.suptitle('Chemoattractant Map');
        ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000);
        ani.save('sim_01210.gif');


def main():
    '''
    runs the scheduler as is, with default parameters except lower step size
    generates 8 plots in the directory at different time intervals.
    '''
    scheduler(steps=300)
    return 0


if __name__ == "__main__":
    main()

