import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import time

def get_acceleration(positions, masses, gravitational_constant, softening_length):
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    dz = z[:, np.newaxis] - z[np.newaxis, :]
    inv_r3 = (dx ** 2 + dy ** 2 + dz ** 2 + softening_length ** 2) ** (-3 / 2)
    np.fill_diagonal(inv_r3, 0)
    ax = - gravitational_constant * np.sum(dx * inv_r3 * masses, axis=1)
    ay = - gravitational_constant * np.sum(dy * inv_r3 * masses, axis=1)
    az = - gravitational_constant * np.sum(dz * inv_r3 * masses, axis=1)
    return np.column_stack((ax, ay, az))

# N-body Simulation Parameters
num_particles = 3
current_time = 0
simulation_end_time = 5
time_step = 0.01
softening_length = 0.000001
gravitational_constant = 1
real_time_plotting = 0
masseq = 1
r3o2 = (3**0.5)/2


# Generate Initial Conditions
np.random.seed(42)
state = np.random.get_state()[1][0]
print(f"Seed = {state}")
total_mass = 1 * num_particles

if masseq == 1:
    masses = total_mass * np.ones(num_particles) / num_particles
elif masseq == 0:
    masses = (0.2 * np.random.randn(num_particles)) + 10
elif masseq == 2:
    masses = np.array([10.8, 9.1])  # alpha centauri AB

#positions = np.random.randn(num_particles, 3) #random
#positions = np.array([[0.,0.,0.],[1.,1.,0]]) #set starts for 2
#positions = np.array([[-1, 0., 0.], [1., 0., 0.], [0., 0., 0.]])  # co-linear 3
positions = np.array([[-0.5,0.,0.],[0,r3o2,0],[0.5,0.,0.]]) #sets the bodies to be equidistant from eachother
#positions = np.array([[0.,0.,0.],[1.,0.,0.],[1.,1.,0.],[0.,1.,0.]]) #sets the bodies to be equidistant from eachother
#positions = np.array([[0,0,0],[1,0,0],[1+c72,s72,0],[0.5,s72+(np.sin(np.radians(36))),0],[-c72,s72,0]]) #pentagon
#positions = np.array([[-0.5,0,0],[0.5,0,0],[1,r3o2,0],[0.5,3**0.5,0],[-0.5,3**0.5,0],[-1,r3o2,0]]) #hexagon
#positions = np.array([[-0.5,0,0],
 #                     [0.5,0,0],
  #                    [0.5+np.cos((2*np.pi)/7),np.sin((2*np.pi)/7),0],
   #                   [0.5+np.cos((2*np.pi)/7)-np.cos((3*np.pi)/7),np.sin((2*np.pi)/7)+np.sin((3*np.pi)/7),0],
    #                  [0,0.5*np.tan(3*np.pi/7),0],
     #                 [-0.5 -np.cos((2*np.pi)/7)+np.cos((3*np.pi)/7),np.sin((2*np.pi)/7)+np.sin((3*np.pi)/7),0],
      #                [-0.5 -np.cos((2*np.pi)/7),np.sin((2*np.pi)/7),0]])
#^heptagon

# pick your fighter for velocites

#velocities = np.random.randn(num_particles, 3) #random
#velocities = np.array([[0.347111, 0.532728, 0.], [0.347111, 0.532728, 0.], [-2*(0.347111), -2*(0.532728), 0.]])  # stable co-linear
velocities = np.array([[0,-1.,0.],[-r3o2,0.5,0.],[r3o2,0.5,0.]]) #triangle
#velocities = np.array([[1.,0.,0.],[0.,1.,0.],[-1.,0.,0.],[0.,-1.,0.]]) #square
#velocities = np.array([[(0.5**0.5),-(0.5**0.5),0.],[(0.5**0.5),(0.5**0.5),0.],[-(0.5**0.5),(0.5**0.5),0.],[-(0.5**0.5),-(0.5**0.5),0.]]) #45
#velocities = np.array([[1,0,0],[c72,s72,0],[-0.5 - c72,np.sin(np.radians(36)),0],[-0.5 - c72,-np.sin(np.radians(36)),0],[c72,-s72,0]]) #pentagon
#velocities = np.array([[1,0,0],[0.5,r3o2,0],[-0.5,r3o2,0],[-1,0,0],[-0.5,-r3o2,0],[0.5,-r3o2,0]])
#velocities = np.array([[1,0,0],
 #                      [np.cos((2*np.pi)/7),np.sin((2*np.pi)/7),0],
  #                     [-np.cos((3*np.pi)/7),np.sin((3*np.pi)/7),0],
   #                    [-np.cos(np.pi/7),np.sin(np.pi/7),0],
    #                   [-np.cos(np.pi/7),-np.sin(np.pi/7),0],
     #                  [-np.cos((3*np.pi)/7),-np.sin((3*np.pi)/7),0],
      #                 [np.cos((2*np.pi)/7),-np.sin((2*np.pi)/7),0]])
#velocities = np.zeros([num_particles,3])

# useful print statements
print(f"Number of bodies: {num_particles}")
print(f"Object masses: {masses}")
print(f"initial positions: {positions}")

# Convert to Center-of-Mass Frame
velocities -= np.mean((masses[:, np.newaxis] * velocities), axis=0) / np.mean(masses)

# Calculate initial gravitational accelerations
acceleration = get_acceleration(positions, masses, gravitational_constant, softening_length)

# Number of timesteps
num_timesteps = int(simulation_end_time / time_step)

# Save energies and particle orbits for plotting trails
saved_positions = np.zeros((num_particles, 3, num_timesteps + 1))
saved_positions[:, :, 0] = positions
all_times = np.arange(num_timesteps + 1) * time_step

# Set the start time for the simulation
start_time = time.time()

# Animation function
def update(frame):
    global positions, velocities, acceleration, current_time
    velocities += acceleration * time_step / 2
    positions += velocities * time_step
    acceleration = get_acceleration(positions, masses, gravitational_constant, softening_length)
    velocities += acceleration * time_step / 2
    current_time += time_step
    saved_positions[:, :, frame] = positions
    ax.clear()
    colours = plt.cm.jet(np.linspace(0, 1, num_particles))
    for i in range(num_particles):
        ax.plot(saved_positions[i, 0, :frame], saved_positions[i, 1, :frame], '.', color=colours[i])
        ax.plot(positions[:, 0], positions[:, 1], 'xm', markersize=14,label = f"Body {i}")

    #ax.set_xlim(-2, 2)  # Set x-axis limits
    #ax.set_ylim(-1, 2)  # Set y-axis limits
    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
    ax.set_title(f"Orbit of {num_particles}-bodies")
    ax.legend()

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(6, 8))

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=num_timesteps, repeat=True, interval=1000)

# End timing the simulation

# Save the animation as a GIF
ani.save(f"n_body_simulation_{num_particles}_time{time.localtime()}.gif", writer='pillow', fps = 60)

# Display the animation
plt.show()

end_time = time.time()
simulation_duration = end_time - start_time


# Print the duration
print(f"Simulation completed in {simulation_duration:.2f} seconds")
print(f"Final positions: {positions}")
# np.savetxt(F"positions_{num_particles}_{total_mass}.csv",positions,delimiter=",")
