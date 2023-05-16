import mesa
import pygame
import random
import math
import time
from AgentType import AgentType

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
        self.space = mesa.space.ContinuousSpace(width, height, False)
        
        for i in range(N):
            x = random.random() * width
            y = random.random() * height
            agent = SystemAgent(i, self, AgentType.FREE, free_agent_behavior(), free_agent_draw(), x, y)
            
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
        # print("Agent " + str(agent.unique_id) + " moved")
        pass
    return behavior

def root_agent_draw():
    def draw(agent, screen):
        pygame.draw.circle(screen, (0, 0, 255), (agent.x, agent.y), 25)
        pygame.draw.circle(screen, (0, 0, 255), (agent.x, agent.y), 5)
    return draw

model = SystemModel(100, 500, 500)
model.run_model()
