import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import time as tim
import matplotlib.animation as animation
from IPython.display import HTML
def update_sphere_positions(state, dt): #good
    #Initialize empty state to be populated with updated state
    new_state = state.copy()
#     print(dt,"dt input")
    for i in range(len(new_state)):
        #Assign the parameters of the sphere to their corresponding variables
        m, r, x, y, vx, vy = new_state[i]
#         print(x,y,vx,vy,"input to update")
        #Update the x and y positions with an Euler step
        x = x+vx*dt
        y = y+vy*dt
        sphere = (m, r, x, y, vx, vy)
#         print(x,y,vx,vy,"output")
        #Populate new_state with updated sphere in state
        new_state[i] = sphere
    return new_state

def find_overlaps_in_state(state): #good
    #Initialize empty list to take indices of colliding spheres
    index = []
    for i in range(len(state)):
        #Assign the parameters of the sphere to their corresponding variables
        m, r, x, y, vx, vy = state[i]
        for n in range(len(state)):
            #Check this sphere for collisions with all spheres in the state
            m2, r2, x2, y2, vx2, vy2 = state[n]
            #Check to see if the sphere overlaps with the other spheres
            # and only add indices of collisions with other spheres
            if n!=i and ((x2-x)**2+(y2-y)**2<=(r+r2)**2):
                index.append((i,n))
    #filter out permutations of indices so we don't double count
    index = list(set(tuple(sorted(l)) for l in index))
    return index

def find_wall_collisions_in_state(state, L): #good
    #Initialize empty list to be populated with tuple containing the index of the sphere colliding with a wall
    # and the wall it is going to collide with
    wallpair = []
    for i in range(len(state)):
        #Assign the parameters of the sphere to their corresponding variables
        m, r, x, y, vx, vy = state[i]
        #Simple checks to see if the sphere will collide with a wall and which wall it will be
        #i.e. if the sphere is less than a radius within range of a wall, it has collided
#         print(x,y,"x and y in find wall")
        if (y+r) >= L:
            wallpair.append((i,"N"))
        if (x+r) >= L:
            wallpair.append((i,"E"))
        if (y-r) <= 0:
            wallpair.append((i,"S"))
        if (x-r) <= 0:
            wallpair.append((i,"W"))
    return wallpair



def compute_speeds(state): #good
    #Initialize empty list to be populated with the speed of each sphere in state
    speeds = []
    for i in range(len(state)):
        #Unpack tuple to find the x and y velocities
        m, r, x, y, vx, vy = state[i]
        #Add the total speed of the sphere to the list
        speeds.append(np.sqrt(vx**2 + vy**2))
    return speeds



def compute_safe_timestep(state,L): #good
    #initialize empty lists to populate with radii and time until collision with each wall
    radii = []
    walltime = []
    safetynet = []
    #Find the speed of the fastest traveling sphere
    vmax = max(compute_speeds(state))
    for i in range(len(state)):
        #Unpack tuple containing sphere parameters
        m, r, x, y, vx, vy = state[i]
        #Append the time until the collision with any one wall
        vel = np.sqrt(vx**2 + vy**2)
        if (L-x-r)<0:
            walltime.append((L-x-r)/vel)
        else: safetynet.append((L-x-r )/vel)
        if (L-y-r)<0:
            walltime.append((L-y-r)/vel)
        else: safetynet.append((L-y-r)/vel)
        if (x-r)<0:
            walltime.append((x-r)/vel)
        else: safetynet.append((x-r)/vel)
        if (y-r)<0:
            walltime.append((y-r)/vel)
        else: safetynet.append((y-r)/vel)
        radii.append(r)
    if walltime != []:
        walltime = min(walltime)
    # Resolves convergence to zero error
    if safetynet != []:
        if walltime<abs(min(safetynet)):
            walltime = abs(min(safetynet))

    ptime = (min(radii)/(2*vmax))
    #Return the smaller of the two timesteps
    return min(ptime,walltime)



def compute_time_to_sphere_collision(state, id0, id1): #good
#     m, r, x, y, vx, vy
# Unpack and assign sphere parameters
    m1, r1, x1, y1, vx1, vy1 = state[id1]
    m0, r0, x0, y0, vx0, vy0 = state[id0]
#Assign relative distance and velocities in unpacked vector form
    dx = x1-x0
    dy = y1-y0
    dvx = vx1-vx0
    dvy = vy1-vy0
    dx2 = dx**2 + dy**2
    dv2 = dvx**2 + dvy**2
    dxdv = dx*dvx + dy*dvy
    R = r0+r1
#Assert the argument of the square root is positive
    assert(dxdv**2>=dv2*(dx2 - R**2))
    dt0 = (1/dv2)*(-dxdv + np.sqrt(dxdv**2 - dv2*(dx2 - R**2)))
    dt1 = (1/dv2)*(-dxdv - np.sqrt(dxdv**2 - dv2*(dx2 - R**2)))
# Properly pick the root based on which of the positive solutions is smaller
    if dt0<=0: dt = dt1
    elif dt1<=0: dt = dt0
    elif dt0<dt1: dt = dt0
    elif dt1<dt0: dt = dt1
#     print(dt,"sphere")
    return dt



def compute_time_to_wall_collision(state, id0, L, wall): #good
    #Unpack and assign the parameters of the sphere
    m, r, x, y, vx, vy = state[id0]
    #Find which wall the sphere is going to collide with, while asserting that it is on a trajectory
    # that collides with the wall
    #Take the absolute value of the distance to the wall so that spheres that 'jumped' over the wall are reflected properly
#     print(x,y,"x,y in compute time")
    if wall == 'N':
        dt = float((L-y-r)/abs(vy))
    elif wall == 'E': 
        dt = float((L-x-r)/abs(vx))
    elif wall == 'S':
        dt = float((y-r)/abs(vy))
    elif wall == 'W':
        dt = float((x-r)/abs(vx))
# return the time until the sphere collides with the wall
#     print(dt,"time to collision")
    return dt



def find_first_collision(state, sphere_hits, wall_hits, L): #test
    #Initialize empty lists for the time until sphere and wall collisions
    t_wall = []
    t_sphere = []
    #For each sphere hit and wall hit found, unpack the values and find the time until each collision occurs
    for i in range(len(wall_hits)):
        indexwall,wall = wall_hits[i] #list of tuple: index and wall 
        t_wall.append(compute_time_to_wall_collision(state,indexwall,L,wall))
    for i in range(len(sphere_hits)):
        indexsphere0,indexsphere1 = sphere_hits[i] #list of two indices that collide
        t_sphere.append(compute_time_to_sphere_collision(state,indexsphere0,indexsphere1))
#initialize the least time until collision to find errors
    minsphere = 'error'
    minwall = 'error'
    
    #Find smallest time until collision for each case
    #If there are both wall and sphere collisions, compare the times in each
    if t_wall != [] and t_sphere!= []:
        minwall = min(t_wall)
        indwall = np.argmin(t_wall)
        minsphere = min(t_sphere)
        indsphere = np.argmin(t_sphere)
        dt = min(minwall,minsphere)
        
# if there are wall collisions and no sphere collisions dt is only the minimum of t_wall
    elif t_wall != [] and t_sphere==[]:
        minwall = min(t_wall)
        dt = minwall
        indwall = np.argmin(t_wall)
        
# if no wall collisions and there are sphere collisions dt is only the minimum of t_sphere
    elif t_sphere!= [] and t_wall == []:
        minsphere = min(t_sphere)
        dt = minsphere 
        indsphere = np.argmin(t_sphere)
        
#Debugging print
    else: print('invalid input for find first collision')
#Assign the collision type based on which type of collision has the smalles time
    if dt==minsphere:
        collision_type = 'sphere'
        collision = sphere_hits[indsphere]
    if dt==minwall:
        collision_type = 'wall'
        collision = wall_hits[indwall]

    return (collision_type, collision, dt)



def resolve_wall_collision(cur_state, id0, wall):
#Initialize a copy of the input state
    res_state = cur_state.copy()
    m, r, x, y, vx, vy = res_state[id0]
#Reflect about the wall of impact if the particle has slipped outside the box
#replace velocity with adjusted (reflected) velocity
    xnew = x
    ynew = y
    if wall == 'N':
        vynew = -1*vy
        vxnew = vx
    if wall == 'E':
        vxnew = -1*vx
        vynew = vy
    if wall == 'S':
        vynew = -1*vy
        vxnew = vx
    if wall == 'W':
        vxnew = -1*vx
        vynew = vy
#Assign the sphere a new trajectory
    res_state[id0] = (m, r, xnew, ynew, vxnew, vynew)
    return res_state



def resolve_sphere_collision(state,id0, id1):
    #Given two spheres p0 and p1 which are currently colliding (exactly in contact), returns 
    #an updated state where the sphere velocities have been updated to account for their collision.
#     m, r, x, y, vx, vy
#Initialize a copy of the input state
    new_state = state.copy()
#Unpack tuple and assign parameters of each sphere to their respective variables
    m0, r0, x0, y0, vx0, vy0 = state[id0]
    m1, r1, x1, y1, vx1, vy1 = state[id1]
#Assign the variables highlighted in kinetic_assign notebook
    bx = (x1-x0)/(r1+r0)
    by = (y1-y0)/(r1+r0)
    bvec = np.array([bx,by])
    v0vec = np.array([vx0,vy0])
    v1vec = np.array([vx1,vy1])
    cx = -by
    cy = bx
    cvec = np.array([cx,cy])
    vb0 = np.dot(v0vec,bvec)
    vb1 = np.dot(v1vec,bvec)
    vc0 = np.dot(v0vec,cvec)
    vc1 = np.dot(v1vec,cvec)
    check = np.array([np.dot(v0vec,bvec)*bvec + np.dot(v0vec,cvec)*cvec])
#Calculate new trajectory
    v0newvec = vb1*bvec+vc0*cvec
    v1newvec = vb0*bvec + vc1*cvec
    vx0new = v0newvec[0]
    vy0new = v0newvec[1]
    vx1new = v1newvec[0]
    vy1new = v1newvec[1]
    
#If the spheres have different masses, run the more complicated calculation
    if m0 != m1:
        vb0new = (m0-m1)*vb0/(m1+m0) + 2*m1*vb1/(m1+m0)
        vb1new = 2*m1*vb0/(m0+m1) - (m0-m1)*vb1/(m1+m0)
        np.array([])
        vx0new = vb0new*bx + vc0*cx
        vy0new = vb0new*by + vc0*cy
        vx1new = vb1new*bx + vc1*cx
        vy1new = vb1new*by + vc1*cy
#Update the state with the new trajectories
    new_state[id0] = (m0, r0, x0, y0, vx0new, vy0new)
    new_state[id1] = (m1, r1, x1, y1, vx1new, vy1new)
    return new_state



def random_initial_state(N_particles, L, v, r, m): #good
#Initialize empty list for state
    state = []
    i=0
    while i < N_particles:
        #Send the particle in a random direction with a total velocity, v
        angle = np.random.rand()*2*np.pi
        vx = v*np.cos(angle)
        vy = v*np.sin(angle)
        #Place the particle randomly in a square of length L
        x = (np.random.rand()*(L-2*r) - r)
        y = (np.random.rand()*(L-2*r) - r)
        #Update a trial state
        temp_state = state + [(m, r, x, y, vx, vy)]
        #If the random sphere doesn't overlap with a wall or another sphere, add this sphere to the state
        if find_overlaps_in_state(temp_state) == [] and find_wall_collisions_in_state(temp_state,L) == []:
            state.append((m, r, x, y, vx, vy))
        #If the random sphere overlaps with a wall or sphere, don't add it to the state and increase the limit on the while
        # loop to account for the skipped sphere
        else: N_particles+=1
        i+=1
    return state

# Same as above but with an object with 100 times the raidius and 100 times the mass of the other particles.
def random_initial_state_brownian(N_particles, L, v, r, m): #good
#Initialize empty list for state
    rb = 10*r
    mb = 10*m
    state = [(mb,rb,L,L,0,0)]
    i=0
    while i < N_particles:
        #Send the particle in a random direction with a total velocity, v
        angle = np.random.rand()*2*np.pi
        vx = v*np.cos(angle)
        vy = v*np.sin(angle)
        #Place the particle randomly in a square of length L
        x = (np.random.rand()*(L-2*r) - r)
        y = (np.random.rand()*(L-2*r) - r)
        #Update a trial state
        temp_state = state + [(m, r, x, y, vx, vy)]
        #If the random sphere doesn't overlap with a wall or another sphere, add this sphere to the state
        if find_overlaps_in_state(temp_state) == [] and find_wall_collisions_in_state(temp_state,L) == []:
            state.append((m, r, x, y, vx, vy))
        #If the random sphere overlaps with a wall or sphere, don't add it to the state and increase the limit on the while
        # loop to account for the skipped sphere
        else: N_particles+=1
        i+=1
    return state



def get_state_at_time(t, times, states):
#Takes output from a simulation
#Define a mutable variable for times
    temptimes = np.array(times)
#Define a placeholder array to identify which time and state in times and states to update with free motion
#marker will contain the difference in time between t and each time in times
#marker also has the same length as times and states
    marker = t - temptimes
    indexfinder = marker
#Find the index in marker at which the desired time occurs
#Mask marker to only contain positive changes in time
    marker = marker[marker>=0]
    marker = min(marker)
    for i in range(len(indexfinder)):
        if indexfinder[i] == marker:
            ind = i
    return update_sphere_positions(states[ind],marker)

def run_API_tests():
    test = None
    failed_tests = []
    test_state = [(1, 1, 5, .5, 10, 10) , (1, 1, .5, 5, 1, 2),
        (1, 1, 5, 9.5, 1, 2), (1, 1, 6, 5, 3, 2),
        (1, 1, 6, 5, 4, 3)]
    if compute_speeds(test_state) != [np.sqrt(200),np.sqrt(5),np.sqrt(5),np.sqrt(13),np.sqrt(25)]:
        test = 'error'
        failed_tests.append('Test #1 error')
        
    
    #should be ~1/28 ~.035
    if abs(1/28 - compute_safe_timestep(test_state,10)) > 1e-3:
        test = 'error'
        failed_tests.append('Test #2 error')

    # Calculated by hand
    if update_sphere_positions(test_state,10) != [(1, 1, 105, 100.5, 10, 10), (1, 1, 10.5, 25, 1, 2), 
                                                     (1, 1, 15, 29.5, 1, 2), (1, 1, 36, 25, 3, 2), (1, 1, 46, 35, 4, 3)]:
        test = 'error'
        failed_tests.append('Test #3 error')

    if compute_time_to_wall_collision(test_state,0,10,'N') != .85:
        test = 'error'
        failed_tests.append('Test #4 error')
        
    sphere_test = [(1,1,2,5,1,0),(1,1,7,5,0,0)]
    if compute_time_to_sphere_collision(sphere_test,0,1) != 3.0:
        test = 'error'
        failed_tests.append('Test #5 error')
        
    wall_sphere = [(1,1,.9,5,1,0)]
    if resolve_wall_collision(wall_sphere, 0 , 'W') != [(1, 1, 0.9, 5, -1, 0)]:
        test = 'error'
        failed_tests.append('Test #6 error')

    if test != None:
        test = failed_tests
    return test

def simulate_gas(initial_state, max_time, L):

    """
    Main loop for hard-sphere kinetic gas project.
    
    Arguments:
    =====
    * initial_state: List of "sphere" tuples giving the initial state.
        A sphere tuple has the form:
            (m, r, x, y, vx, vy)
        where m is the mass, r is the radius, x and y are the coordinates, and
        vx and vy are the velocity.
    * max_time: The maximum amount of time to allow the system to evolve for.
        (Units depend on how you implement the API - be consistent!)
    * L: Side length of the square (LxL) box in which the simulation is run.
    
    Returns:
    =====
    (times, states): Two lists containing the times at which any collisions occurred,
        and the state of the system immediately after each collision.
    
    Example usage:
    =====
    >> state0 = random_initial_state(10, 200, 10, 2, 1)
    >> (times, states) = simulate_gas(state0, 60, 10)
    
    Now you can run measurements on `states`, or use the plot_state()
    function below to plot states after each collision, or even make an
    animation with get_state_at_time().
    
    """
    
    times = [0]
    states = [initial_state]
    eps = 1e-3  # "Overshoot" factor for moving past safe timestep - this should be small!

    time = 0
    cur_state = initial_state
    tStart = tim.time()
    while time < max_time:
        dt = compute_safe_timestep(cur_state, L)
#         print("Safe timestep = %g" % dt)
 
        
#         for item in cur_state:
#             print("State:" ,item[2], item[3])
        
        # Try advancing state
        proposed = update_sphere_positions(cur_state, dt)
        # Check for wall or inter-sphere collisions
        sphere_hits = find_overlaps_in_state(proposed)
        wall_hits = find_wall_collisions_in_state(proposed, L)
        
#         print(" %d sphere hits, %d wall hits" % (len(sphere_hits), len(wall_hits)))
        
        if len(sphere_hits) == 0 and len(wall_hits) == 0:
            # Nothing interesting happened.  Keep state and move on
            time += dt
            cur_state = proposed
#             print("Nothing interesting happened at time", time)
            continue
            
        # Find first collision and what kind it was
        collision_type, first_collision, dt = find_first_collision(cur_state, sphere_hits, wall_hits, L)
#         print("%s collision, (%s, %s), dt = %g" % (collision_type, first_collision[0], first_collision[1], dt))
        # Handle collision
        proposed = update_sphere_positions(cur_state, dt * (1+eps))

        if collision_type == "sphere":
            id0, id1 = first_collision

            # Resolve elastic collision and plug back in to state
            proposed = resolve_sphere_collision(proposed, id0, id1)

        elif collision_type == "wall":
            id0, wall0 = first_collision

            # Update state to account for wall collision
            
            proposed = resolve_wall_collision(proposed, id0, wall0)


        # Save new state and time
        time += dt
        cur_state = proposed
        times.append(time)
        states.append(cur_state)
        #print the time elapsed in real time for simulation to run
#         print("T = {}s elapsed realtime".format(round(tim.time()-tStart, 3)))
#         print("Saving new state at time", time)
    
    print("Simulation completed after T = {}s.".format(round(tim.time()-tStart, 3)))
    # Save final state
    times.append(time)
    states.append(cur_state)
    return times, states

def plot_state_frame(ax, state, color='blue'):
    # Extract positions
    r = []
    x = []
    y = []
    for i in range(len(state)):
        m0, r0, x0, y0, vx0, vy0 = state[i]
        r.append(r0)
        x.append(x0)
        y.append(y0)
    # plt.scatter doesn't work for this -- scatter can't get size in data units, only "points"
    circles = []
    for nn in range(len(state)):
        circles.append(ax.add_artist(patches.Circle(xy=(x[nn], y[nn]), radius=r[nn], facecolor=color)))
        
    return circles

def plot_state(state, color="#003366"):
    # Extract positions
    r = []
    x = []
    y = []
    for m0, r0, x0, y0, vx0, vy0 in state:
        r.append(r0)
        x.append(x0)
        y.append(y0)
        
    # plt.scatter doesn't work for this -- scatter can't get size in data units, only "points"
    for nn in range(len(state)):
        plt.gca().add_artist(patches.Circle(xy=(x[nn], y[nn]), radius=r[nn], facecolor=color))
        
    return
def animate_simulation(times,states,L,max_time,numframes=50):
    fig, ax = plt.subplots()
    ax.set_xlim(0,L)
    ax.set_ylim(0,L)
    frames = []
    for t in np.linspace(0,max_time,numframes):
        frames.append(plot_state_frame(ax, get_state_at_time(t, times, states)))
    ani = animation.ArtistAnimation(fig, frames, interval=100, blit=True, repeat=False)
    return ani,fig