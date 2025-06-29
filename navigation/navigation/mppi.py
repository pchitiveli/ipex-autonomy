import torch
import numpy as np
import matplotlib.pyplot as plt
import math
from gen_spiral_path import PredefinedPath
from tqdm import tqdm
import matplotlib.animation as animation
from pathlib import Path

class MPPI_controller():
    def __init__(self, 
                 horizons,
                 rollouts,
                 dt, 
                 temperature, 
                 noise_std, 
                 init_pos, 
                 path_deviation_const = 1,
                 camera_aim_const = 1, 
                 rock_collision_const = 2):
        
        # setting up cuda
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print("Using device:", self.device)
        
        # setting parameters
        self.horizons = horizons
        self.rollouts = rollouts
        self.dt = dt
        self.temperature = temperature
        self.noise_std = noise_std
        self.init_pos = torch.tensor(init_pos, device=self.device, dtype=torch.float32)
        self.path_deviation_const = path_deviation_const
        self.camera_aim_const = camera_aim_const
        self.rock_collision_const = rock_collision_const
        
        # basic constants
        self.u_dim = 2 # input control [linear, angular]
        self.state_dim = 3 # state [x, y, theta]

        ##### setting up the initial state #####
        # current state of the robot
        self.state = torch.zeros(self.state_dim, device=self.device, dtype=torch.float32)
        
        # stores the states of the rollouts
        self.states = torch.zeros((self.rollouts, self.horizons, self.state_dim), device=self.device, dtype=torch.float32)
        
        # stores the best control sequence
        self.u_seq = torch.zeros((self.horizons, self.u_dim), device=self.device, dtype=torch.float32)
        
        # stores the noise
        self.noise = torch.zeros((self.rollouts, self.horizons, self.u_dim), device=self.device, dtype=torch.float32)
        
        # stores the control sequences
        self.u = torch.zeros((self.rollouts, self.horizons, self.u_dim), device=self.device, dtype=torch.float32)

        # stores the cost of each rollout
        self.cost = torch.zeros(self.rollouts, device=self.device, dtype=torch.float32)

        # for the enviornemnt
        self.cell_size = 5 # size of the cell in the grid in centimeters
        self.grid_size = 3 * 60
        self.padding_grids = 10

        self.path_cost_matrix = torch.zeros((self.grid_size, self.grid_size), device=self.device, dtype=torch.float32)
        self.rock_cost_matrix = torch.zeros((self.grid_size, self.grid_size), device=self.device, dtype=torch.float32)
        self.direction_cost_matrix = torch.zeros((self.grid_size, self.grid_size), device=self.device, dtype=torch.float32)
        self.combined_cost_matrix = torch.zeros((self.grid_size, self.grid_size), device=self.device, dtype=torch.float32)

        self.path = []
        self.camera_target = torch.tensor([250, 0], device=self.device, dtype=torch.float32)
        self.rocks = []

        self.path_generator = PredefinedPath(self.grid_size)
        self.path_generator.generate_path(init_pos)
        self.path_lookahead = 30

        self.rock_preference_sizing = 3

        self.kernal_size = 11
        self.kernal_right = torch.zeros((self.kernal_size, self.kernal_size), dtype=torch.float32)
        for i in range(self.kernal_size):
            for j in range(self.kernal_size):
                self.kernal_right[i, j] = -1 + i * 2/(self.kernal_size - 1)

        self.kernal_right = torch.nn.functional.normalize(self.kernal_right, dim=0)
        self.kernal_right = self.kernal_right.unsqueeze(0).unsqueeze(0).to(device=self.device)

        self.kernal_down = torch.zeros((self.kernal_size, self.kernal_size), dtype=torch.float32)
        for i in range(self.kernal_size):
            for j in range(self.kernal_size):
                self.kernal_down[i, j] = -1 + j * 2/(self.kernal_size - 1)

        self.kernal_down = torch.nn.functional.normalize(self.kernal_down, dim=0)
        self.kernal_down = self.kernal_down.unsqueeze(0).unsqueeze(0).to(device=self.device)

        self.padding = torch.zeros((self.grid_size, self.grid_size), device=self.device, dtype=torch.float32)

        # NOTE: do we need to do update_total_cost()

    def _gaussian(self, size, sigma, height = 1):
        """
        Generates a gaussian distribution
        Args:
            size (Tuple): size of the gaussian
            sigma (float): standard deviation of the gaussian
            height (float): height of the gaussian

        Returns:
            torch.Tensor: gaussian distribution
        """
        ax = torch.linspace(-(size - 1) / 2., (size - 1) / 2., size)
        gauss = torch.exp(-0.5 * torch.square(ax) / sigma**2)
        kernel = torch.outer(gauss, gauss)
        kernel /= torch.sum(kernel)
        kernel *= height / kernel[size//2, size//2]
        return kernel.to(device=self.device)
    
    def _clamp(self, value, minimum, maximum):
        """
        Clamps a value between a minimum and maximum

        Args:
            value (float): value to clamp
            minimum (float): minimum value
            maximum (float): maximum value

        Returns:
            float: clamped value
        """
        return max(minimum, min(value, maximum))
        
    def set_path(self, path):
        """
        Sets the path for the robot to follow

        Updates the `path_cost_matrix` and `combined_cost_matrix` based on the new path.

        Args:
            path (List): list of points in the path
        """
        if (self.path == path):
            return
        
        # print(path)
        # get the path in terms of grid cells
        # print(path)
        points = [(p[1] // self.cell_size, p[0] // self.cell_size) for p in path]
        # points = sorted(set(points))
        # print("Path points:", points)
        # print(points)

        path_len = len(points)

        self.path_cost_matrix = torch.zeros((self.grid_size, self.grid_size), device=self.device, dtype=torch.float32)

        for i, point in enumerate(points):
            # this is all in terms of grid cells
            x, y = point
            radius = 5 * self.cell_size * 3
            size = (2 * radius + 4, 2 * radius + 4)
            clamped_size = [self._clamp(x - radius - 2, 0, self.grid_size), self._clamp(x + radius + 2, 0, self.grid_size),
                            self._clamp(y - radius - 2, 0, self.grid_size), self._clamp(y + radius + 2, 0, self.grid_size)]
            actual_size = self.path_cost_matrix[
                clamped_size[0]:clamped_size[1], clamped_size[2]:clamped_size[3]].shape

            gaussian_start_x = max(0, radius + 2 - x)  # Offset if x is near 0
            gaussian_end_x = gaussian_start_x + actual_size[0]

            gaussian_start_y = max(0, radius + 2 - y)  # Offset if y is near 0
            gaussian_end_y = gaussian_start_y + actual_size[1]

            gaussian = self._gaussian(size[0], 9, (i + 1)/path_len + 0.4)[gaussian_start_x:gaussian_end_x, gaussian_start_y:gaussian_end_y]
            sliced = self.path_cost_matrix[clamped_size[0]:clamped_size[1], clamped_size[2]:clamped_size[3]]
            # print(clamped_size[1] - clamped_size[0], clamped_size[3] - clamped_size[2])
            # print(gaussian.shape)

            self.path_cost_matrix[clamped_size[0]:clamped_size[1], clamped_size[2]:clamped_size[3]] = torch.maximum(gaussian, sliced).to(device=self.device)
        self.path_cost_matrix *= -1
        self.path_cost_matrix += 1.2
        self.path = path
        self.update_total_cost()

    def add_rock(self, rock):
        """
        Adds a rock to the controller

        Updates the rock matrix such that the rock is surrounded by a gaussian distribution
        for the cost function based on the height and the radius of the rock

        Args:
            rock (tuple): (x, y, height, radius)
        """
        x, y, height, radius = rock
        x = max(0,  x // self.cell_size)
        y = max(0,  y // self.cell_size)
        radius = max(1, radius // self.cell_size)
        
        if ([x, y, height] in self.rocks):
            return
        else:
            self.rocks.append([x, y, height])

        # sets the surrounding cells to a gaussian distribution based on the height and radius of the rock
        size = (radius + 6, radius + 6)
        clamped_size = [self._clamp(x - radius//2 - 3, 0, self.grid_size), self._clamp(x + radius//2 + 3, 0, self.grid_size),
                        self._clamp(y - radius//2 - 3, 0, self.grid_size), self._clamp(y + radius//2 + 3, 0, self.grid_size)]
        actual_size = self.rock_cost_matrix[
            clamped_size[0]:clamped_size[1], clamped_size[2]:clamped_size[3]].shape

        gaussian_start_x = max(0, radius//2 + 3 - x)  # Offset if x is near 0
        gaussian_end_x = gaussian_start_x + actual_size[0]

        gaussian_start_y = max(0, radius//2 + 3- y)  # Offset if y is near 0
        gaussian_end_y = gaussian_start_y + actual_size[1]



        # self.rock_cost_matrix[clamped_size[0]:clamped_size[1], clamped_size[2]:clamped_size[3]] = torch.max(
        #     gaussian[gaussian_start_x:gaussian_end_x, gaussian_start_y:gaussian_end_y], 
        #     self.rock_cost_matrix[clamped_size[0]:clamped_size[1], clamped_size[2]:clamped_size[3]]).to(device=self.device)
        self.rock_cost_matrix[clamped_size[0]:clamped_size[1], clamped_size[2]:clamped_size[3]] += self._gaussian(size[0], radius, height)[gaussian_start_x:gaussian_end_x, gaussian_start_y:gaussian_end_y]
        
        squeezed = self.rock_cost_matrix.unsqueeze(0).unsqueeze(0)

        output_right = torch.nn.functional.conv2d(squeezed, self.kernal_right, padding=self.kernal_size//2)
        output_down = torch.nn.functional.conv2d(squeezed, self.kernal_down, padding=self.kernal_size//2)

        output = (output_down + output_right).to(device=self.device)

        self.padding = torch.zeros((self.grid_size, self.grid_size), device=self.device, dtype=torch.float32)
        self.padding = torch.where(mppi.rock_cost_matrix == 0, output.squeeze(), torch.zeros_like(output.squeeze()))
        self.padding = torch.where(self.padding > 0, torch.full_like(self.padding, 0.3), self.padding)
        self.padding = torch.where(self.padding < 0, torch.full_like(self.padding, -0.3), self.padding)

        self.update_total_cost()
    
    def update_total_cost(self):
        """
        Updates the total cost matrix based on the path and rock cost matrices
        Takes into account the constants
        """

        self.combined_cost_matrix = self.path_cost_matrix * self.path_deviation_const + self.rock_cost_matrix * self.rock_collision_const + self.padding

    def generate_noise(self):
        """
        Generates noise for the control sequences
        """
        self.noise = torch.normal(0, self.noise_std, (self.rollouts, self.horizons, self.u_dim), device=self.device)
        self.noise[:, :, 0] += 0.2 # adding a bias to the linear velocity
        self.noise[:, :, 0] *= 100 # converting linear velocity to cm/s
        self.noise[:, :, 0] = torch.clamp(self.noise[:, :, 0], 0, 30) # clamping linear velocity to the maximum speed of the rover

        self.noise[:, :, 1] *= 5 # adding more spread to the angular velocity

    def sample_trajectories(self):
        """
        Generates `self.rollout` trajectories
        Updates 'self.states' and `self.u` of the controller
        """
        # setting the linear velocity otherwise it would be just the max speed of the rover
        self.u[:, :, 0] = self.noise[:, :, 0]
        # adding the angular velocity to bias it towards what is already good
        # self.u[:, :, 1] += self.noise[:, :, 1]

        # self.u[:, :, 1] = self.noise[:, :, 1] + self.u[:, :, 1] * 0.7

        self.u[:, :, 1] = self.noise[:, :, 1] + self.u_seq[:, 1] * 0.5

        self.simulate_rollouts()
        self.compute_costs()

    def simulate_rollouts(self):
        """
        Simulates the rollouts based on the current state and control sequences
        """
        self.states = torch.zeros((self.rollouts, self.horizons, self.state_dim), dtype=torch.float32, device=self.device)
        self.states[:, 0, :] = self.state

        linear_velocity = self.u[:, :, 0]
        angular_velocity = self.u[:, :, 1]

        theta = torch.cumsum(angular_velocity * self.dt, dim=1) + self.state[2]

        dx = linear_velocity * torch.cos(theta) * self.dt
        dy = linear_velocity * torch.sin(theta) * self.dt

        self.states[:, :, 0] = torch.cumsum(dx, axis=1) + self.state[0]
        self.states[:, :, 1] = torch.cumsum(dy, axis=1) + self.state[1]
        self.states[:, :, 2] = theta
        out_of_bounds = (self.states[:, :, 0] < 0) | (self.states[:, :, 0] >= self.grid_size * self.cell_size) | \
                        (self.states[:, :, 1] < 0) | (self.states[:, :, 1] >= self.grid_size * self.cell_size)
        
        self.states[out_of_bounds] = torch.tensor([-1, -1, -1], device=self.device, dtype=torch.float32)

    def compute_costs(self):
        """
        Computes the costs for each rollout based on the current state and control sequences
        """
        oob_mask = (self.states == torch.tensor([-1, -1, -1], device=self.device)).all(dim=2)

        x = self.states[:, :, 0] / self.cell_size
        y = self.states[:, :, 1] / self.cell_size
        x = x.type(torch.int32)
        y = y.type(torch.int32)

        safe_x = torch.where(oob_mask, torch.zeros_like(x), x)
        safe_y = torch.where(oob_mask, torch.zeros_like(y), y)

        path_cost = self.combined_cost_matrix[safe_y, safe_x]

        penalty = 10

        path_cost = torch.where(oob_mask, torch.tensor(penalty, device=self.device, dtype=path_cost.dtype), path_cost)

        # gives you the average cost of each rollout
        self.cost = torch.sum(path_cost, dim=1) / self.horizons


    def set_state(self, state):
        """
        Sets the state of the robot

        Args:
            state (array): state of the robot [x, y, theta]
        """
        self.state = torch.tensor(state, device=self.device, dtype=torch.float32)

        locs = self.path_generator.get_path(self.path_lookahead)
        last_loc = 0
        # dists = []

        for i, loc in enumerate(locs):
            # dists.append(math.sqrt((state[0] - loc[0]) ** 2 + (state[1] - loc[1]) ** 2))
            if math.sqrt((state[0] - loc[0]) ** 2 + (state[1] - loc[1]) ** 2) < 75:
                last_loc = i + 1

        self.path_generator.remove_path(last_loc)
        # print(min(dists))
        if last_loc != 0:
            print("Removed path:", last_loc)

        # dist_first = math.sqrt((state[0] - next_loc[1]) ** 2 + (state[1]  - next_loc[0]) ** 2)
        # dist_second = math.sqrt((state[0]  - second_loc[1]) ** 2 + (state[1] - second_loc[0]) ** 2)

        # print(dist_first, dist_second)

        # if dist_first < 50:
            # self.path_generator.remove_path(1)
        # if dist_second < dist_first:
            # self.path_generator.remove_path(1)

        self.set_path(self.path_generator.get_path(self.path_lookahead))

    def update_control_seq(self):
        """
        Updates the control sequence based on the current state and control sequences
        """
        weights = torch.exp(-self.cost / self.temperature)
        weights /= torch.sum(weights)
        self.u_seq = torch.sum(weights[:, None, None] * self.u, dim=0)

    def get_optimal_control(self):
        """
        Returns the optimal control sequence based on the current state and control sequences

        Returns:
            numpy array: optimal control sequence
        """
        control = self.u_seq[0, :]
        control[0] = self._clamp(control[0], 0, 48)
        control[1] = self._clamp(control[1], -math.pi, math.pi)
        self.u_seq = torch.roll(self.u_seq, -1, 0)

        if control.cpu().numpy()[0] == 0:
            print("________________________________")
            print(self.u_seq.cpu().numpy())
            print(self.noise.cpu().numpy())
            print(self.cost.cpu().numpy())
            print(self.states.cpu().numpy())
            print(self.u.cpu().numpy())
            self.states = torch.zeros((self.rollouts, self.horizons, self.state_dim), device=self.device, dtype=torch.float32)
            self.u_seq = torch.zeros((self.horizons, self.u_dim), device=self.device, dtype=torch.float32)
            self.noise = torch.zeros((self.rollouts, self.horizons, self.u_dim), device=self.device, dtype=torch.float32)
            self.u = torch.zeros((self.rollouts, self.horizons, self.u_dim), device=self.device, dtype=torch.float32)

            

        return control.cpu().numpy()
    
    def run(self):
        """
        Runs the MPPI controller
        """
        self.generate_noise()
        self.sample_trajectories()
        self.update_control_seq()
        return self.get_optimal_control()

# if __name__ == "__main__":
#     mppi = MPPI_controller(horizons=100, rollouts=100, dt=.1, temperature=0.01, noise_std=0.1, init_pos=[0, 0])
#     mppi.set_state([0, 0, 0])

#     rocks = np.loadtxt("rockmap.txt", dtype=np.int32)
#     rocks = torch.from_numpy(rocks)
#     # for i in range(rocks.shape[0]):
#     #     for j in range(rocks.shape[1]):
#     #         if rocks[i, j] == 1:
#     #             mppi.add_rock((i * mppi.cell_size * 3, j * mppi.cell_size * 3, 1, 15))

#     plt.ion()
#     path_travelled = np.zeros(3, dtype=np.float32)
#     plt.imshow(mppi.combined_cost_matrix.cpu().numpy(), cmap='viridis', interpolation='nearest', origin='lower')
    
#     plt.colorbar()
#     for i in tqdm(range(10_000)):
#         control = mppi.run()
#         plt.clf()
#         plt.imshow(mppi.combined_cost_matrix.cpu().numpy(), cmap='viridis', interpolation='nearest', origin='lower', )
#         plt.gca().set_xlim([0, mppi.grid_size])
#         plt.gca().set_ylim([0, mppi.grid_size])
#         # plt.imshow(mppi.path_cost_matrix.cpu().numpy(), cmap='viridis', interpolation='nearest', origin='lower')
        
#         min_cost = torch.min(mppi.cost).cpu().numpy()
#         max_cost = torch.max(mppi.cost).cpu().numpy()
#         norm = plt.Normalize(vmin=min_cost, vmax=max_cost)
#         cmap = plt.cm.hot

#         for path, cost in zip(mppi.states.cpu().numpy(), mppi.cost.cpu().numpy()):
#             path /= mppi.cell_size
#             color = cmap(norm(cost))
#             plt.plot(path[:, 0], path[:, 1], alpha=0.2, color=color)
        
#         current_state = mppi.state.cpu().numpy()
#         states = mppi.state.cpu().numpy()
#         for control in mppi.u_seq.cpu().numpy():
#             current_state[2] += control[1] * mppi.dt
#             current_state[0] += control[0] * math.cos(current_state[2]) * mppi.dt
#             current_state[1] += control[0] * math.sin(current_state[2]) * mppi.dt
#             states = np.vstack((states, current_state))

#         new_state = mppi.state.cpu().numpy()
#         new_state[2] += control[1] * mppi.dt
#         new_state[0] += control[0] * math.cos(new_state[2]) * mppi.dt
#         new_state[1] += control[0] * math.sin(new_state[2]) * mppi.dt

#         path_travelled = np.vstack((path_travelled, new_state/mppi.cell_size))

#         plt.plot(new_state[0]/mppi.cell_size, new_state[1]/mppi.cell_size, 'ro', markersize=10)
#         plt.plot(states[:, 0]/mppi.cell_size, states[:, 1]/mppi.cell_size, 'b-')
#         # print(mppi.path_generator.get_path(20))
#         # plt.plot([(p[0]/15, p[1]/15) for p in mppi.path_generator.get_path(20)])
#         # plt.plot(mppi.path_generator.get_path(20)[:, 1]/15, mppi.path_generator.get_path(20)[:, 0]/15, 'r-')
#         plt.plot(path_travelled[:, 0], path_travelled[:, 1], 'g-')

#         for i in range(int(min(max(0, new_state[1]//(mppi.cell_size * 3) - 10), mppi.grid_size)), int(min(max(0, new_state[1]//(mppi.cell_size * 3) + 10), mppi.grid_size))):
#             for j in range(int(min(max(0, new_state[0]//(mppi.cell_size * 3) - 10), mppi.grid_size)), int(min(max(0, new_state[0]//(mppi.cell_size * 3) + 10), mppi.grid_size))):
#                 if rocks[i, j] == 1:
#                     mppi.add_rock((i * mppi.cell_size * 3, j * mppi.cell_size * 3, 1, 15))


#         plt.pause(1e-10)
#         mppi.set_state(new_state)
#     plt.show()

if __name__ == "__main__":
    mppi = MPPI_controller(horizons=100, rollouts=100, dt=.1, temperature=0.01, noise_std=0.1, init_pos=[0, 0])
    mppi.set_state([0, 0, 0])

    rocks = np.loadtxt("rockmap.txt", dtype=np.int32)
    rocks = torch.from_numpy(rocks)
    # for i in range(rocks.shape[0]):
    #     for j in range(rocks.shape[1]):
    #         if rocks[i, j] == 1:
    #             mppi.add_rock((i * mppi.cell_size * 3, j * mppi.cell_size * 3, 1, 15))

    # Set up for both live display and video recording
    save_video = True
    video_filename = "mppi_simulation3.mp4"
    video_fps = 350
    total_frames = 10000  # Change this to control simulation length
    
    # Create figure for plotting
    fig, ax = plt.subplots(figsize=(10, 8))
    path_travelled = np.zeros(3, dtype=np.float32)
    
    # Store frames for animation
    frames = []
    
    def update_plot(frame_num):
        ax.clear()
        
        # Run MPPI controller
        control = mppi.run()
        print(control)
        
        # Plot environment
        ax.imshow(mppi.combined_cost_matrix.cpu().numpy(), cmap='viridis', interpolation='nearest', origin='lower')
        ax.set_xlim([0, mppi.grid_size])
        ax.set_ylim([0, mppi.grid_size])
        
        min_cost = torch.min(mppi.cost).cpu().numpy()
        max_cost = torch.max(mppi.cost).cpu().numpy()
        norm = plt.Normalize(vmin=min_cost, vmax=max_cost)
        cmap = plt.cm.hot
        
        # Plot potential paths
        for path, cost in zip(mppi.states.cpu().numpy(), mppi.cost.cpu().numpy()):
            path /= mppi.cell_size
            color = cmap(norm(cost))
            ax.plot(path[:, 0], path[:, 1], alpha=0.2, color=color)
        
        # Get optimal trajectory
        current_state = mppi.state.cpu().numpy()
        states = mppi.state.cpu().numpy()
        for u in mppi.u_seq.cpu().numpy():
            current_state[2] += u[1] * mppi.dt
            current_state[0] += u[0] * math.cos(current_state[2]) * mppi.dt
            current_state[1] += u[0] * math.sin(current_state[2]) * mppi.dt
            states = np.vstack((states, current_state))
        
        # Update state
        new_state = mppi.state.cpu().numpy()
        new_state[2] += control[1] * mppi.dt
        new_state[0] += control[0] * math.cos(new_state[2]) * mppi.dt
        new_state[1] += control[0] * math.sin(new_state[2]) * mppi.dt
        
        # Record path traveled
        global path_travelled
        path_travelled = np.vstack((path_travelled, new_state/mppi.cell_size))
        
        # Plot current position and paths
        ax.plot(new_state[0]/mppi.cell_size, new_state[1]/mppi.cell_size, 'ro', markersize=10)
        ax.plot(states[:, 0]/mppi.cell_size, states[:, 1]/mppi.cell_size, 'b-')
        ax.plot(path_travelled[:, 0], path_travelled[:, 1], 'g-')
        
        # Set title with frame number and control values
        ax.set_title(f"Frame: {frame_num} - Linear v: {control[0]:.2f}, Angular v: {control[1]:.2f}")
        
        global rocks
        for i in range(int(min(max(0, new_state[1]//(mppi.cell_size * 3) - 10), mppi.grid_size)), int(min(max(0, new_state[1]//(mppi.cell_size * 3) + 10), mppi.grid_size))):
            for j in range(int(min(max(0, new_state[0]//(mppi.cell_size * 3) - 10), mppi.grid_size)), int(min(max(0, new_state[0]//(mppi.cell_size * 3) + 10), mppi.grid_size))):
                if rocks[i, j] == 1:
                    mppi.add_rock((i * mppi.cell_size * 3, j * mppi.cell_size * 3, 1, 15))

        # Update state for next iteration
        mppi.set_state(new_state)
        
        return ax,
    
    if save_video:
        print(f"Creating animation with {total_frames} frames...")
        ani = animation.FuncAnimation(fig, update_plot, frames=range(total_frames), 
                                      blit=False, repeat=False)
        
        # Check if FFmpeg is available
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=video_fps, metadata=dict(artist='MPPI Simulation'), bitrate=1800)
        
        print(f"Saving video to {video_filename}...")
        ani.save(video_filename, writer=writer)
        print(f"Video saved successfully to {video_filename}")
        
        # Display the final result
        plt.close()
        print("Finished!")
    else:
        # Live interactive display as before
        plt.ion()
        for i in tqdm(range(total_frames)):
            update_plot(i)
            plt.pause(1e-10)
        plt.ioff()
        plt.show()