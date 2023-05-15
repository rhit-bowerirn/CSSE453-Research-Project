import mesa
import pygame
import random
import math
import time

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
            if not(dist == 0) and other_robot.role == "root":
                mag = lj_magnitude(dist, 40, 50)
                total_dx += mag * dx / dist
                total_dy += mag * dy / dist
    return (total_dx, total_dy)


class SystemAgent(mesa.Agent):
    def __init__(self, unique_id, model, role, start_x, start_y):
        super().__init__(unique_id, model)
        self.role = role
        self.x = start_x
        self.y = start_y

    def step(self):
        vec = lj_vector(self, self.model.agents)
        self.x += vec[0]
        self.y += vec[1]
        if not(self.model.space.out_of_bounds((self.x, self.y))):
            self.model.space.move_agent(self, (self.x, self.y))

    def draw(self, screen):
        pygame.draw.circle(screen, (255, 0, 0), (self.x, self.y), 5)

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
            agent = SystemAgent(i, self, "free", x, y)

            self.agents.append(agent)
            self.schedule.add(agent)
            self.space.place_agent(agent, (x, y))

        root = SystemAgent(N, self, "root", 250, 250)
        self.agents.append(root)
        self.schedule.add(root)
        self.space.place_agent(root, (250, 250))

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

model = SystemModel(100, 500, 500)
model.run_model()
