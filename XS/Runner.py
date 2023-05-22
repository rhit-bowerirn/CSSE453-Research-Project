import mesa
import pygame
import random
import math
import time
import numpy as np
from AgentType import AgentType
from Color import Color
find = 0

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

        self.screen = pygame.display.set_mode((width, height))
        self.screen.fill((255, 255, 255))

        self.num_agents = N
        self.agents = []
        self.width = width
        self.height = height
        self.schedule = mesa.time.RandomActivation(self)
        self.space = mesa.space.ContinuousSpace(width, height, True)
        
        for i in range(N-1):
            x = random.random() * width
            y = random.random() * height

            agent = SystemAgent(i, self, AgentType.SCOUT, scout_agent_behavior(), scout_agent_draw(), x, y)
            
            self.agents.append(agent)
            self.schedule.add(agent)
            self.space.place_agent(agent, (x, y))

        root_agent = SystemAgent(N-1, self, AgentType.ROOT, root_agent_behavior(), root_agent_draw(), width // 2, height // 2)
        self.agents.append(root_agent)
        self.schedule.add(root_agent)
        self.space.place_agent(root_agent, (width // 2, height // 2))

        target = SystemAgent(N, self, AgentType.TARGET, target_agent_behavior(), target_agent_draw(), 10, 10)
        self.agents.append(target)
        self.schedule.add(target)
        self.space.place_agent(target, (10, 10))

        self.running = True
    
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
def scout_agent_behavior(comm_radius = 40, min_strength = 1, b = 5, speed = 5, rnd = .05):
    def behavior(agent):
        neighbors = agent.model.space.get_neighbors((agent.x,agent.y),comm_radius,include_center=False)
        re = searching(agent, neighbors)
        if re == -1:
            return
        elif re == 1:
            # TODO find target
            return
        else:
            # TODO explore

    # iterate through each neighbor,
    # if no neighbors, then return -1
    # if find target, then return 1
    # else return 0
    def searching(agent, neighbors):
        neighbors = agent.model.space.get_neighbors((agent.x,agent.y),comm_radius,include_center=False)
        if(len(neighbors)==0):
            return -1
        for neighbor in neighbors:
            if (neighbor.role == AgentType.TARGET):
                return 1
                print(neighbor.role)
        return 0
    
    def request():
    return behavior

def scout_agent_draw(comm_radius = 40):
    def draw(agent,screen):
        neighbors = agent.model.space.get_neighbors((agent.x,agent.y),comm_radius,include_center=False)
        pygame.draw.circle(screen, Color.SCOUT.value, (agent.x, agent.y), 5)
        for neighbor in neighbors:
            pygame.draw.line(screen,Color.LINE.value,(agent.x,agent.y),(neighbor.x,neighbor.y))
    return draw

# networker robots
def networker_agent_behavior(comm_radius = 40, min_strength = 1, b = 5, speed = 5, rnd = .05):
    def behavior(agent):
        neighbors = agent.model.space.get_neighbors((agent.x,agent.y),comm_radius,include_center=False)
        strn, grad = comm_gradient(agent,neighbors)
        if len(neighbors) == 0:
            strn = 0
            grad = np.array([agent.x-250,agent.y-250])
            grad = grad/np.sqrt((grad*grad).sum())
        direction = grad*(strn-min_strength)
        direction = speed*direction
        new_pos = direction + np.array([agent.x,agent.y])
        if not agent.model.space.out_of_bounds(new_pos):
            agent.model.space.move_agent(agent,new_pos)
            agent.x = new_pos[0]
            agent.y = new_pos[1]
            return
    def comm_gradient(agent, neighbors):
        total_strength = 0
        total_gradient = np.array([0,0])
        for neighbor in neighbors:
            dx = agent.x - neighbor.x
            dy = agent.y - neighbor.y
            strength = b/(b+dx*dx+dy*dy)-b/(b+comm_radius)
            total_strength += strength
            dsdx = -2*(dx)*b/((b+dx*dx+dy*dy)**2)
            dsdy = -2*(dy)*b/((b+dx*dx+dy*dy)**2)
            total_gradient = total_gradient + np.array([dsdx,dsdy])
        return total_strength, total_gradient
    # def request 
    return behavior
def networker_agent_draw(comm_radius = 40):
    def draw(agent,screen):
        neighbors = agent.model.space.get_neighbors((agent.x,agent.y),comm_radius,include_center=False)
        pygame.draw.circle(screen, Color.SCOUT.value, (agent.x, agent.y), 5)
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

model = SystemModel(100, 1000, 1000)
model.run_model()
