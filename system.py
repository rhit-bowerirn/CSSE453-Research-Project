import mesa
import pygame
import random
import math
import time
import numpy as np
from AgentType import AgentType
import cProfile
from numba import jit, cuda
import io


from ParallelActivation import ParallelActivation


comm_radius = 50
vision_radius = 20
max_seen_strength = 1
min_seen_radius = 1
isToroidal = False
width = 1000
height = 1000


strength = []

pheremoneMap = {'seen':np.zeros((width,height))}

def lj_magnitude(dist, lj_target, lj_epsilon):
    return -(lj_epsilon/dist) * ((lj_target/dist)**4-(lj_target/dist)**2)

def lj_vector(robot, other_robots):
    total_dx = 0
    total_dy = 0
    for other_robot in other_robots:
        if other_robot is not robot:
            dx = other_robot.pos[0] - robot.pos[0]
            dy = other_robot.pos[1] - robot.pos[1]
            dist = math.sqrt(dx**2 + dy**2)
            if not(dist == 0) and other_robot.role.value == "root":
                mag = lj_magnitude(dist, 40, 50)
                total_dx += mag * dx / dist
                total_dy += mag * dy / dist
    return (total_dx, total_dy)


class SystemAgent(mesa.Agent):
    def __init__(self, unique_id, model, role: AgentType, behavior_func, draw_func):
        super().__init__(unique_id, model)
        self.role = role
        self.behavior_func = behavior_func
        self.draw_func = draw_func
        self.parent = None
        self.children = []
        self.checked_connection = False
        self.isConnected = False
        # self.pheremoneMap = {'seen':np.zeros((self.model.space.width,self.model.space.height))}
        self.data = {'scout_follower':False,'scout_data':False,'root_data':False, 'strength':0}
        
    def step(self):
        self.behavior_func(self)
        self.checked_connection = False
        neighbors = self.model.space.get_neighbors((self.pos[0],self.pos[1]),comm_radius,include_center=False)
        # self.pheremoneMap['seen'] = self.pheremoneMap['seen']*.999
        # for neighbor in neighbors:
            # neighbor.recievePheremone(self.pheremoneMap)
        

    def send(self):
        pass

    def recieve(self):
        pass

    def getPheremone(self,pheremone,x,y):
        x = int(x)
        y = int(y)
        if isToroidal:
            x = x%self.model.space.width
            y = y%self.model.space.height
        if self.model.space.out_of_bounds((x,y)):
            return max_seen_strength
        # return self.pheremoneMap[pheremone][x][y]
        global pheremoneMap
        return pheremoneMap[pheremone][x][y]

    
    def getConnections(self):
        return self.model.space.get_neighbors((self.pos[0],self.pos[1]),comm_radius,include_center=False)

    
    # def recievePheremone(self,pheremoneMap):
    #     for key in pheremoneMap.keys():
    #         self.pheremoneMap[key] = np.maximum(self.pheremoneMap[key],pheremoneMap[key])

    def isConnectedToRoot(self):
        if self.role == AgentType.ROOT:
            return True
        if self.checked_connection:
            return self.isConnected
        self.checked_connection = True
        self.isConnected = False
        neighbors = self.getConnections()
        for neighbor in neighbors:
            if neighbor.isConnectedToRoot():
                self.isConnected = True
                return True
        return False
    
    def isConnectedToScout(self):
        if self.role == AgentType.SCOUT:
            return True
        if self.checked_connection:
            return self.isConnected
        self.checked_connection = True
        self.isConnected = False
        neighbors = self.getConnections()
        for neighbor in neighbors:
            if neighbor.isConnectedToScout():
                self.isConnected = True
                return True
        return False


    def draw(self, screen):
        self.draw_func(self, screen)

class SystemModel(mesa.Model):
    def __init__(self, N, width, height):
        # self.schedule = ParallelActivation(self) # TODO Depending on where this is in the init list, different errors happen
        # self.schedule = RandomActivation(self) 

        pygame.init()

        self.screen = pygame.display.set_mode((width, height))
        self.screen.fill((255, 255, 255))

        self.num_agents = N
        self.agents = []
        self.width = width
        self.height = height
        self.schedule = mesa.time.RandomActivation(self)
        self.space = mesa.space.ContinuousSpace(width, height, isToroidal)

        # self.space = mesa.space.ContinuousSpace(width, height, False)

        self.clock = pygame.time.Clock()
        self.total_time = self.clock.get_time()
        self.last_print_time = self.total_time
        self.one_second_frame_count = 0
        
        for i in range(N):
            x = random.random() * width
            y = random.random() * height
            agent = SystemAgent(i, self, AgentType.FREE, scout_agent_behavior(), scout_agent_draw())
            
            self.agents.append(agent)
            self.schedule.add(agent)
            self.space.place_agent(agent, (x, y))

        self.root_agent = SystemAgent(N, self, AgentType.ROOT, root_agent_behavior(), root_agent_draw())
        self.agents.append(self.root_agent)
        self.schedule.add(self.root_agent)
        self.space.place_agent(self.root_agent, (width // 2, height // 2))

        self.running = True
    
    def step(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                self.quit()
        self.schedule.step()
        self.update_display()
        self.total_time += self.clock.get_time()
        if self.total_time - self.last_print_time >= 1000:
            print(f"FPS: {self.one_second_frame_count}")
            self.one_second_frame_count = 0
            self.last_print_time = self.total_time
        self.one_second_frame_count += 1
        self.clock.tick()   

    def update_display(self):
        # surf = pygame.surfarray.make_surface(self.root_agent.pheremoneMap['seen']*255/max_seen_strength)
        global pheremoneMap
        surf = pygame.surfarray.make_surface(pheremoneMap['seen']*255/max_seen_strength)
        # pheremoneMap['seen'] = pheremoneMap['seen']*.995
        self.screen.blit(surf, (0,0))
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
        agent.pos[0] += vec[0]
        agent.pos[1] += vec[1]
    return behavior

def free_agent_draw():
    def draw(agent, screen):
        pygame.draw.circle(screen, (255, 0, 0), (agent.pos[0], agent.pos[1]), 5)
    return draw

# Root robots
def root_agent_behavior():
    def behavior(agent):
        pass
    return behavior

def root_agent_draw():
    def draw(agent, screen):
        pygame.draw.circle(screen, (0, 0, 255), (agent.pos[0], agent.pos[1]), 5)
    return draw

# Scout robots
def normalize(vec):
    norm = np.linalg.norm(vec)
    if norm != 0:
        return vec/norm
    return vec

def network_agent_behavior(speed = 5, home = np.array([250,250]), last_root = np.array([250,250]), last_scout = np.array([250,250])):
    def behavior(agent : SystemAgent):
        pos = np.array(agent.pos)
        neighbors = agent.getConnections()
        withScout = False
        for n in neighbors:
            if n.role == AgentType.SCOUT:
                withScout = True
                agent.data['scout_data'] = True
                agent.data['root_data'] = False
                if len(n.getConnections()) < 2:
                    agent.data['scout_follower'] = True
                else:
                    agent.data['scout_follower'] = False
            elif n.role == AgentType.NETWORKER:
                if n.data['root_data']:
                    agent.data['root_data'] = True
            elif n.role == AgentType.ROOT:
                agent.data['root_data'] = True
                agent.data['scout_data'] = False
    return behavior

def network_agent_draw():
    def draw(agent,screen):
        neighbors = agent.getConnections()
        pygame.draw.circle(screen, (128,0,255), (agent.pos[0], agent.pos[1]), 5)
        for neighbor in neighbors:
            pygame.draw.line(screen,(128,128,128),(agent.pos[0],agent.pos[1]),(neighbor.pos[0],neighbor.pos[1]))
    return draw


def scout_agent_behavior(speed = 5, relative_strengths = (.98,.01,.01), min_strength = .5 , home = np.array([width//2,height//2]), last_connection = np.array([width//2,height//2]), vis_rad = vision_radius, seen_strength = max_seen_strength, min_rad = min_seen_radius):
    def behavior(agent : SystemAgent):
        pos = np.array(agent.pos)
        neighbors = agent.getConnections()
        phstr, phgrad = seen_gradient(agent)
        strength.append(phstr)
        agent.data['strength'] = phstr
        phgrad = normalize(phgrad)
        commstr, commgrad = comm_gradient(agent,neighbors)
        commgrad = commgrad * (commstr - min_strength)
        commgrad = normalize(commgrad)
        randgrad = np.random.normal(size=2)
        randgrad = normalize(randgrad)
        new_pos = relative_strengths[0]*phgrad
        new_pos = relative_strengths[1]*commgrad + new_pos
        new_pos = relative_strengths[2]*randgrad + new_pos
        if len(neighbors)==0:
            if np.linalg.norm(last_connection-pos) < 5:
                last_connection[0] = home[0]
                last_connection[1] = home[1]
            new_pos = new_pos + normalize(last_connection-pos)
        else:
            last_connection[0] = pos[0]
            last_connection[1] = pos[1]
        # if not agent.updateConnection():
        #         new_pos = new_pos + normalize(home-pos)
        new_pos = normalize(new_pos)
        new_pos *= speed
        new_pos = new_pos + pos
        
        if isToroidal or not agent.model.space.out_of_bounds(new_pos):
            agent.model.space.move_agent(agent,new_pos)
            # gpu_update_seen(agent.pheremoneMap['seen'],agent.pos[0],agent.pos[1],vis_rad,min_rad,seen_strength)
            # update_seen(agent)
            global pheremoneMap
            gpu_update_seen(pheremoneMap['seen'],agent.pos[0],agent.pos[1],vis_rad,min_rad,seen_strength)

            return

    # def update_seen(agent):
    #     A = (min_seen_radius*min_seen_radius+vis_rad*vis_rad)
    #     A = seen_strength*min_seen_radius*min_seen_radius*A
    #     B = vis_rad*vis_rad*min_seen_radius*min_seen_radius
    #     C = seen_strength*min_seen_radius*min_seen_radius/(vis_rad*vis_rad)
    #     xmax = agent.model.space.width
    #     ymax = agent.model.space.height
    #     x = int(agent.pos[0])
    #     y = int(agent.pos[1])
    #     for i in range(vision_radius):
    #         i2 = i*i
    #         for j in range(vision_radius):
    #             strength = A/(vis_rad*vis_rad*(i2+j*j)+B)-C
    #             x1 = x-i
    #             x2 = x+i
    #             y1 = y-j
    #             y2 = y+j
    #             if(x1>=0 and x1<xmax):
    #                 if(y1>=0 and y1<ymax):
    #                     agent.pheremoneMap['seen'][x1][y1] = max(strength,agent.pheremoneMap['seen'][x1][y1])
    #                 if(y2>=0 and y2<ymax):
    #                     agent.pheremoneMap['seen'][x1][y2] = max(strength,agent.pheremoneMap['seen'][x1][y2])
    #             if(x2>=0 and x2<xmax):
    #                 if(y1>=0 and y1<ymax):
    #                     agent.pheremoneMap['seen'][x2][y1] = max(strength,agent.pheremoneMap['seen'][x2][y1])
    #                 if(y2>=0 and y2<ymax):
    #                     agent.pheremoneMap['seen'][x2][y2] = max(strength,agent.pheremoneMap['seen'][x2][y2])

    def seen_gradient(agent, samples = 40):
        R2 = 0
        total_strength = 0
        total_gradient = np.array([0,0])
        for i in range(samples):
            dx = random.normalvariate(sigma=vis_rad)
            dy = random.normalvariate(sigma=vis_rad)
            x = int(agent.pos[0]+dx)
            y = int(agent.pos[1]+dy)
            dx = x-agent.pos[0]
            dy = y-agent.pos[1]
            stren = agent.getPheremone('seen',x,y)
            total_strength += stren*(dx*dx+dy*dy) #weight the average by how close they are to the robot
            R2 += (dx*dx+dy*dy)
            try:
                dsdx = (seen_strength - stren)/dx #we know we have seen the current spot as well as possible
            except:
                dsdx = (seen_strength - stren)
            try:
                dsdy = (seen_strength - stren)/dy
            except:
                dsdy = (seen_strength - stren)
            total_gradient = total_gradient + np.array([dsdx,dsdy])
        total_strength /= R2
        total_gradient /= samples
        return total_strength, total_gradient
    
    def comm_gradient(agent, neighbors):
        total_strength = 0
        total_gradient = np.array([0,0])
        for neighbor in neighbors:
            dx = agent.pos[0] - neighbor.pos[0]
            dy = agent.pos[1] - neighbor.pos[1]
            strength = 1/(1+dx*dx+dy*dy)-1/(1+comm_radius*comm_radius)
            total_strength += strength
            dsdx = -2*(dx)*1/((1+dx*dx+dy*dy)**2)
            dsdy = -2*(dy)*1/((1+dx*dx+dy*dy)**2)
            total_gradient = total_gradient + np.array([dsdx,dsdy])
        return total_strength, total_gradient
    return behavior

def scout_agent_draw():
    def draw(agent,screen):
        # neighbors = agent.getConnections()
        if agent.data['strength'] < .5:
            pygame.draw.circle(screen, (128,0,255), (agent.pos[0], agent.pos[1]), 5)
        else: 
            pygame.draw.circle(screen, (255,0,128), (agent.pos[0], agent.pos[1]), 5) #these agents are inside a well seen blob and should be used as free or networkers
        # for neighbor in neighbors:
            # pygame.draw.line(screen,(128,128,128),(agent.pos[0],agent.pos[1]),(neighbor.pos[0],neighbor.pos[1]))
    return draw


@jit(target_backend='cuda')
def gpu_update_seen(arr,x,y,r,m,i):
    A = (m*m+r*r)
    A = i*m*m*A
    B = r*r*m*m
    C = i*m*m/(r*r)
    # for i in range(len(arr)):
    for i in range(max(0,int(x-vision_radius)),min(width,int(x+vision_radius))):
        dx2 = (i-x)*(i-x)
        # for j in range(len(arr[i])):
        for j in range(max(0,int(y-vision_radius)),min(height,int(y+vision_radius))):
            dy2 = (j-y)*(j-y)
            strength = A/(r*r*(dy2+dx2)+B)-C
            arr[i][j] = max(strength,arr[i][j])


model = SystemModel(50, width, height)
# model.run_model()
# cProfile.run('model.run_model()',sort='tottime')

import pstats

pr = cProfile.Profile()
pr.enable()
try:
    model.run_model()
except:
    pass
pr.disable()

s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
ps.print_stats()

with open('test.txt', 'w+') as f:
    f.write(s.getvalue())

print(np.mean(strength))
print(np.percentile(strength,0))
print(np.percentile(strength,10))
print(np.percentile(strength,20))
print(np.percentile(strength,30))
print(np.percentile(strength,40))
print(np.percentile(strength,50))
print(np.percentile(strength,60))
print(np.percentile(strength,70))
print(np.percentile(strength,80))
print(np.percentile(strength,90))
print(np.percentile(strength,100))