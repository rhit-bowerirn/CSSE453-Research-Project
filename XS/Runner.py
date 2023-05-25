import mesa
import pygame
import random
import math
import time
import numpy as np
from AgentType import AgentType
from Color import Color
import os
NUM_SCOUTS = 10
NUM_NETWORKERS = 88

# Globals 
move_rate = 10
scout_radius = 40
networker_radius = 50
target_x = None
target_y = None
root_x = None
root_y = None
# 2 variables for the scouts\

find = 0
near_target = 0
# 2 variable for the networkers
path = 0
follow = 0


def lj_magnitude(dist, lj_target, lj_epsilon):
    return -(lj_epsilon / dist) * ((lj_target / dist)**4 - (lj_target / dist)**.5)

def lj_vector(robot, other_robots):
    total_dx = 0
    total_dy = 0
    for other_robot in other_robots:
        if other_robot is not robot:
            dx = other_robot.x - robot.x
            dy = other_robot.y - robot.y
            dist = math.sqrt(dx**2 + dy**2)
            if not(dist == 0) and other_robot.role.value == "root":
                mag = lj_magnitude(dist, 25, 100)
                total_dx += mag * dx / dist
                total_dy += mag * dy / dist
    return (total_dx, total_dy)

def obstacle_avoidance(robot, other_robots, safe_dist = 25, k = .4):
    total_dx = 0
    total_dy = 0
    for other_robot in other_robots:
        if other_robot is not robot:
            dx = other_robot.x - robot.x
            dy = other_robot.y - robot.y
            dist = math.sqrt(dx**2 + dy**2)
            if(dist < safe_dist):
                mag = k * (dist - safe_dist)
                total_dx += mag * dx / dist
                total_dy += mag * dy / dist
    return(total_dx, total_dy)



class SystemAgent(mesa.Agent):
    def __init__(self, unique_id, model, role: AgentType, behavior_func, draw_func, start_x, start_y):
        super().__init__(unique_id, model)
        self.role = role
        self.behavior_func = behavior_func
        self.draw_func = draw_func
        self.x = start_x
        self.y = start_y
        self.parent = None
        self.children = []
        self.toRoot = False
        # This is for scout that have agent to follow
        self.followed = True
        
    def step(self):
        self.behavior_func(self)

    def send(self):
        pass

    def recieve(self):
        pass

    def draw(self, screen):
        self.draw_func(self, screen)

class SystemModel(mesa.Model):
    def __init__(self, N, width, height):

        pygame.init()
        global root_x, root_y
        root_x = width//2
        root_y = height//2

        self.screen = pygame.display.set_mode((width, height))
        self.screen.fill((255, 255, 255))

        self.num_agents = N
        self.agents = []
        self.width = width
        self.height = height
        self.schedule = mesa.time.RandomActivation(self)
        self.space = mesa.space.ContinuousSpace(width, height, True)

        root_agent = SystemAgent(0, self, AgentType.ROOT, root_agent_behavior(), root_agent_draw(), root_x, root_y)
        self.agents.append(root_agent)
        self.schedule.add(root_agent)
        self.space.place_agent(root_agent, (root_x, root_y))

        target = SystemAgent(1, self, AgentType.TARGET, target_agent_behavior(), target_agent_draw(), 10, 10)
        self.agents.append(target)
        self.schedule.add(target)
        self.space.place_agent(target, (10, 10))

        self.running = True

        scout_process = []
        for process in range(NUM_SCOUTS):
            # pid = os.fork()
            # if pid:
            #     scout_process.append(pid)
            # else:
                # x = random.random() * width
                # y = random.random() * height

            agent = SystemAgent(process+2, self, AgentType.SCOUT, scout_agent_behavior(), scout_agent_draw(), root_x, root_y)
                
            self.agents.append(agent)
            self.schedule.add(agent)
            self.space.place_agent(agent, (root_x, root_y))

        for networker in range(NUM_NETWORKERS):
            agent = SystemAgent(networker+2+NUM_SCOUTS, self, AgentType.NETWORKER, networker_agent_behavior(), networker_agent_draw(),root_x, root_y)
            self.agents.append(agent)
            self.schedule.add(agent)
            self.space.place_agent(agent, (root_x, root_y))

    
    def step(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                self.quit()
        self.schedule.step()
        self.update_display()

    def update_display(self):
        self.screen.fill((255, 255, 255))
        self.draw_agents()
        pygame.display.flip()

    def draw_agents(self):
        for agent in self.agents:
            agent.draw(self.screen)

    def quit(self):
        pygame.quit()

# Scout robots
def scout_agent_behavior(comm_radius = scout_radius, min_strength = 1, b = 5, speed = 5, rnd = .05):
    def behavior(agent):
        global find, target_x,target_y
        neighbors = agent.model.space.get_neighbors((agent.x,agent.y),comm_radius,include_center=False)
        re, target = searching(agent, neighbors)
        agent.followed = False

        if re == -1:
            # No neighbors, lost connection
            return
        elif re == 1:
            find = 1
            target_x = target.x
            target_y = target.y
            new_pos = move_to_root(agent) if near_target(agent,neighbors) else move_to_target(agent)
            if not agent.model.space.out_of_bounds(new_pos):
                agent.model.space.move_agent(agent,new_pos)
                agent.x = new_pos[0]
                agent.y = new_pos[1]
                return
        else:
            # TODO explore
            new_pos = explore(agent)
            if not agent.model.space.out_of_bounds(new_pos):
                agent.model.space.move_agent(agent,new_pos)
                agent.x = new_pos[0]
                agent.y = new_pos[1]
                return
            return

    # iterate through each neighbor,
    # if no neighbors, then return -1
    # if find target, then return 1
    # else return 0
    def searching(agent, neighbors):
        # neighbors = agent.model.space.get_neighbors(pos=np.array([agent.x,agent.y]),radius=comm_radius,include_center=False)
        if(len(neighbors)==1 and neighbors[0].unique_id == agent.unique_id):
            return -1, None
        for neighbor in neighbors:
            if (neighbor.role == AgentType.TARGET):
                return 1, neighbor
        return 0, None
    
    # Function that check if near target, return 1 if so, 0 otherwise
    def near_target(agent,neighbors):
        for neighbor in neighbors:
            if neighbor.role == AgentType.TARGET:
                return 1
        return 0


    # Function that get new position when target is found
    def move_to_target(agent):
        total = target_x+target_y
        diff_x = agent.x - target_x
        diff_y = agent.y - target_y
        vec = np.array([target_x - agent.x, target_y - agent.y])
        scale = move_rate / (diff_x **2 + diff_y ** 2)
        new_pos = np.array([agent.x, agent.y]) + vec * scale
        return new_pos

    # Funtion that get new position when back to root
    def move_to_root(agent):
        total = root_x+root_y
        diff_x = agent.x - root_x
        diff_y = agent.y - root_y
        vec = np.array([root_x - agent.x, root_y - agent.y])
        scale = move_rate / (diff_x **2 + diff_y ** 2)
        new_pos = np.array([agent.x, agent.y]) + vec * scale
        return new_pos

    # Function that generate a new direction
    def explore(agent):
        random_angle  = random.uniform(0, 2*math.pi)
        vec = np.array([move_rate * math.sin(random_angle), move_rate*math.cos(random_angle)])
        new_pos = np.array([agent.x, agent.y])+vec
        return new_pos
        
    def request():
        pass
    return behavior

def scout_agent_draw(comm_radius = 40):
    def draw(agent,screen):
        pygame.draw.circle(screen, Color.SCOUT.value, (agent.x, agent.y), 5)
    return draw

# networker robots
def networker_agent_behavior(comm_radius = networker_radius, min_strength = 1, b = 5, speed = 5, rnd = .05):
    def behavior(agent):
        neighbors = agent.model.space.get_neighbors((agent.x,agent.y),comm_radius,include_center=False)
        # strn, grad = comm_gradient(agent,neighbors)
        # if len(neighbors) == 0:
        #     strn = 0
        #     grad = np.array([agent.x-250,agent.y-250])
        #     grad = grad/np.sqrt((grad*grad).sum())
        # direction = grad*(strn-min_strength)
        # direction = speed*direction
        # new_pos = direction + np.array([agent.x,agent.y])
        new_pos = follow(agent, neighbors)
        if not agent.model.space.out_of_bounds(new_pos):
            agent.model.space.move_agent(agent,new_pos)
            agent.x = new_pos[0]
            agent.y = new_pos[1]
            return
    def follow(agent, neighbors):
        for neighbor in neighbors:
            diff_x = neighbor.x - agent.x
            diff_y = neighbor.y - agent.y
            dist = math.sqrt(diff_x**2 + diff_y**2)
            if dist > scout_radius-10 and neighbor.followed == False:
                neighbor.followed = True
                scale = move_rate/(dist)
                vec = np.array([diff_x,diff_y])
                new_pos = np.array([agent.x, agent.y])+vec*scale
                neighbor.follow = True
                return new_pos
        return np.array([-1,-1])

    # def comm_gradient(agent, neighbors):
    #     total_strength = 0
    #     total_gradient = np.array([0,0])
    #     for neighbor in neighbors:
    #         dx = agent.x - neighbor.x
    #         dy = agent.y - neighbor.y
    #         strength = b/(b+dx*dx+dy*dy)-b/(b+comm_radius)
    #         total_strength += strength
    #         dsdx = -2*(dx)*b/((b+dx*dx+dy*dy)**2)
    #         dsdy = -2*(dy)*b/((b+dx*dx+dy*dy)**2)
    #         total_gradient = total_gradient + np.array([dsdx,dsdy])
    #     return total_strength, total_gradient
    # def request 
    return behavior
def networker_agent_draw(comm_radius = 40):
    def draw(agent,screen):
        neighbors = agent.model.space.get_neighbors((agent.x,agent.y),comm_radius,include_center=False)
        pygame.draw.circle(screen, Color.NETWORKER.value, (agent.x, agent.y), 5)
        for neighbor in neighbors:
            pygame.draw.line(screen,Color.LINE.value,(agent.x,agent.y),(neighbor.x,neighbor.y))
    return draw

# Root robots
def root_agent_behavior():
    def behavior(agent):
        pass
    return behavior

def root_agent_draw():
    def draw(agent, screen):
        pygame.draw.circle(screen, Color.ROOT.value, (agent.x, agent.y), 5)
    return draw

# target, do nothing
def target_agent_behavior():
    def behavior(agent):
        pass
    return behavior

def target_agent_draw():
    def draw(agent, screen):
        pygame.draw.circle(screen, Color.TARGET.value, (agent.x, agent.y), 5)
    return draw

model = SystemModel(2+NUM_NETWORKERS+NUM_SCOUTS, 1000, 1000)
model.run_model()
