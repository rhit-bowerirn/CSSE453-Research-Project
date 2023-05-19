import mesa
import pygame
import random
import math
import time
import numpy as np
from AgentType import AgentType
import cProfile

comm_radius = 40
vision_radius = 10

# pheremoneMap = {'seen':np.zeros((500,500))}

def lj_magnitude(dist, lj_target, lj_epsilon):
    return -(lj_epsilon/dist) * ((lj_target/dist)**4-(lj_target/dist)**2)

def lj_vector(robot, other_robots):
    total_dx = 0
    total_dy = 0
    for other_robot in other_robots:
        if other_robot is not robot:
            dx = other_robot.x - robot.x
            dy = other_robot.y - robot.y
            dist = math.sqrt(dx**2 + dy**2)
            if not(dist == 0) and other_robot.role.value == "root":
                mag = lj_magnitude(dist, 40, 50)
                total_dx += mag * dx / dist
                total_dy += mag * dy / dist
    return (total_dx, total_dy)


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
        self.checked_connection = False
        self.isConnected = False
        self.pheremoneMap = {'seen':np.zeros((500,500))}
        
    def step(self):
        self.behavior_func(self)
        self.checked_connection = False
        self.pheremoneMap['seen'] = self.pheremoneMap['seen']*.95

    def send(self):
        pass

    def recieve(self):
        pass

    # def getPheremone(self, pheremone, x, y, polled):
    #     x = int(x)
    #     y = int(y)
    #     if x<0 or x>499 or y<0 or y>499:
    #         return 0
    #     level = self.pheremoneMap[pheremone][int(x)][int(y)]
    #     if self in polled:
    #         return level
    #     polled.add(self)
    #     neighbors = self.model.space.get_neighbors((self.x,self.y),comm_radius,include_center=False)
    #     for neighbor in neighbors:
    #         level = max(level,neighbor.getPheremone(pheremone,x,y,polled))
    #     self.pheremoneMap[pheremone][int(x)][int(y)] = level
    #     return level

    def getPhenemoe(self,pheremone,x,y):
        x = int(x)
        y = int(y)
        if x<0 or x>499 or y<0 or y>499:
            return 0
        return self.pheremoneMap[pheremone][x][y]
    
    def recievePheremone(self,pheremoneMap):
        for key in pheremoneMap.keys():
            self.pheremoneMap[key] = np.maximum(self.pheremoneMap[key],pheremoneMap[key])

    def updateConnection(self):
        if self.role == AgentType.ROOT:
            return True
        if self.checked_connection:
            return self.isConnected
        self.checked_connection = True
        self.isConnected = False
        neighbors = self.model.space.get_neighbors((self.x,self.y),comm_radius,include_center=False)
        for neighbor in neighbors:
            if neighbor.updateConnection():
                self.isConnected = True
                return True
        return False


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
        self.space = mesa.space.ContinuousSpace(width, height, False)
        
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
        vec = lj_vector(agent, agent.model.agents)
        agent.x += vec[0]
        agent.y += vec[1]
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
 
def scout_agent_behavior(min_strength = 1, b = 5, speed = 5, rnd = .05):
    def behavior(agent : SystemAgent):
        neighbors = agent.model.space.get_neighbors((agent.x,agent.y),comm_radius,include_center=False)
        phstr, phgrad = seen_gradient(agent)
        direction = 10*phgrad
        strn, grad = comm_gradient(agent,neighbors)
        direction = direction + grad*(strn-min_strength)
        direction = direction + rnd*np.array([random.normalvariate(),random.normalvariate()])
        if not agent.updateConnection():
            home = np.array([agent.x-250,agent.y-250])
            home = home/np.sqrt((home*home).sum())
            direction = direction - home
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
            update_seen(agent)
            return
        print('oob')

    def update_seen(agent):
        threshold = b/(b+vision_radius*vision_radius)
        for i in range(-vision_radius,vision_radius):
            i2 = i*i
            for j in range(-vision_radius,vision_radius):
                strength = b/(b+i2+j*j)-threshold
                # strength = b/(b+i*i+j*j)-b/(b+vision_radius*vision_radius)
                x = int(i+agent.x)
                y = int(j+agent.y)
                if (x>=0 and x<=499 and y>=0 and y<=499):
                    # pheremoneMap['seen'][x][y] = max(strength,pheremoneMap['seen'][x][y])
                    agent.pheremoneMap['seen'][x][y] = max(strength,agent.pheremoneMap['seen'][x][y])

    def seen_gradient(agent, samples = 4):
        total_strength = 0
        total_gradient = np.array([0,0])
        for i in range(samples):
            dx = random.normalvariate(sigma=vision_radius)
            dy = random.normalvariate(sigma=vision_radius)
            # stren = agent.getPheremone('seen',agent.x+dx,agent.y+dy,set())
            x = int(agent.x+dx)
            y = int(agent.y+dy)
            stren = 0
            if not(x<0 or x>499 or y<0 or y>499):
                stren = agent.pheremoneMap['seen'][x][y]
            total_strength = stren*b/(b+dx*dx+dy*dy)
            dsdx = stren*-2*(dx)*b/((b+dx*dx+dy*dy)**2)
            dsdy = stren*-2*(dy)*b/((b+dx*dx+dy*dy)**2)
            total_gradient = total_gradient + np.array([dsdx,dsdy])
        return total_strength, total_gradient
    
    def comm_gradient(agent, neighbors):
        total_strength = 0
        total_gradient = np.array([0,0])
        for neighbor in neighbors:
            dx = agent.x - neighbor.x
            dy = agent.y - neighbor.y
            strength = b/(b+dx*dx+dy*dy)-b/(b+comm_radius*comm_radius)
            total_strength += strength
            dsdx = -2*(dx)*b/((b+dx*dx+dy*dy)**2)
            dsdy = -2*(dy)*b/((b+dx*dx+dy*dy)**2)
            total_gradient = total_gradient + np.array([dsdx,dsdy])
        return total_strength, total_gradient
    return behavior
def scout_agent_draw():
    def draw(agent,screen):
        neighbors = agent.model.space.get_neighbors((agent.x,agent.y),comm_radius,include_center=False)
        pygame.draw.circle(screen, (128,0,255), (agent.x, agent.y), 5)
        for neighbor in neighbors:
            pygame.draw.line(screen,(128,128,128),(agent.x,agent.y),(neighbor.x,neighbor.y))
    return draw

model = SystemModel(100, 500, 500)
cProfile.run('model.run_model()')
