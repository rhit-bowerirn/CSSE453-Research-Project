import mesa
import pygame
import random
import math
import time
import numpy as np
from AgentType import AgentType

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
        
        for i in range(N):
            x = random.random() * width
            y = random.random() * height

            agent = SystemAgent(i, self, AgentType.FREE, scout_agent_behavior(), scout_agent_draw(), x, y)
            
            self.agents.append(agent)
            self.schedule.add(agent)
            self.space.place_agent(agent, (x, y))

        root_agent = SystemAgent(N, self, AgentType.ROOT, root_agent_behavior(), root_agent_draw(), width // 2, height // 2)
        self.agents.append(root_agent)
        self.schedule.add(root_agent)
        self.space.place_agent(root_agent, (width // 2, height // 2))

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

# Free robots
def free_agent_behavior():
    def behavior(agent):
        lj = lj_vector(agent, agent.model.agents)
        obst = obstacle_avoidance(agent, agent.model.agents)
        agent.x += lj[0] + obst[0]
        agent.y += lj[1] + obst[1]
        agent.model.space.move_agent(agent, (agent.x, agent.y))
    return behavior

def free_agent_draw():
    def draw(agent, screen):
        pygame.draw.circle(screen, (255, 0, 0), (agent.x, agent.y), 5)
    return draw

# Root robots
def root_agent_behavior():
    def behavior(agent):
        pass
    return behavior

def root_agent_draw():
    def draw(agent, screen):
        pygame.draw.circle(screen, (0, 0, 255), (agent.x, agent.y), 5)
    return draw

# Scout robots
 
def scout_agent_behavior(comm_radius = 40, min_strength = 1, b = 5, speed = 5, rnd = .05):
    def behavior(agent : SystemAgent):
        neighbors = agent.model.space.get_neighbors((agent.x,agent.y),comm_radius,include_center=False)
        strn, grad = comm_gradient(agent,neighbors)
        if len(neighbors) == 0:
            strn = 0
            grad = np.array([agent.x-250,agent.y-250])
            grad = grad/np.sqrt((grad*grad).sum())
        direction = grad*(strn-min_strength)
        direction = direction + rnd*np.array([random.normalvariate(),random.normalvariate()])
        direction = speed*direction
        new_pos = direction + np.array([agent.x,agent.y])
        # direction = (grad[0] * (strn - min_strength),grad[1] * (strn - min_strength))
        # direction = (random.random()+direction[0],random.random()+direction[1])
        # direction = (direction[0],direction[1])
        # new_pos = (agent.x+direction[0],agent.y+direction[1])
        if not agent.model.space.out_of_bounds(new_pos):
            agent.model.space.move_agent(agent,new_pos)
            agent.x = new_pos[0]
            agent.y = new_pos[1]
            return
        print('oob')
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
    return behavior

def scout_agent_draw(comm_radius = 40):
    def draw(agent,screen):
        neighbors = agent.model.space.get_neighbors((agent.x,agent.y),comm_radius,include_center=False)
        pygame.draw.circle(screen, (128,0,255), (agent.x, agent.y), 5)
        for neighbor in neighbors:
            pygame.draw.line(screen,(128,128,128),(agent.x,agent.y),(neighbor.x,neighbor.y))
    return draw

model = SystemModel(100, 600, 600)
model.run_model()
