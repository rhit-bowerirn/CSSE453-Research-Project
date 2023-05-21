import mesa
import pygame
import random
import math
import time
from Root import Root
from Worker import Worker
from Networker import Networker

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
            agent = SystemAgent(self,i, Networker, x, y)
            
        #     self.agents.append(agent)
        #     self.schedule.add(agent)
        #     self.space.place_agent(agent, (x, y))

        # root_agent = SystemAgent(N, self, AgentType.ROOT, root_agent_behavior(), root_agent_draw(), width // 2, height // 2)
        # self.agents.append(root_agent)
        # self.schedule.add(root_agent)
        # self.space.place_agent(root_agent, (width // 2, height // 2))

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

class Agent(mesa.Agent):
    def __init__(self, unique_id, model, start_x,start_y):
        super().__init__(unique_id,model)
        self.x  = start_x
        self.y  = start_y

class SystemAgent(mesa.Agent):
    def __init__(self, unique_id, model, role, start_x, start_y):
        super().__init__(unique_id, model)
        # self.role = role
        self.x = start_x
        self.y = start_y
        model.behavior()

    def step(self):
        vec = lj_vector(self, self.model.agents)
        self.x += vec[0]
        self.y += vec[1]
        if not(self.model.space.out_of_bounds((self.x, self.y))):
            self.model.space.move_agent(self, (self.x, self.y))

    def draw(self, screen):
        pygame.draw.circle(screen, (255, 0, 0), (self.x, self.y), 5)
def free_agent_behavior():
    pass
def free_agent_draw():
    pass

root = Root(1)
print(root.ID)
# server = ModularServer(MyModel, [canvasvis, graphvis], name="My Model") server.launch()


model = SystemModel(100, 500, 500)
model.run_model()
