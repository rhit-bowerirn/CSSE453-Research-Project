from concurrent.futures import ProcessPoolExecutor, wait
from typing import Optional

from mesa.model import Model
from mesa.time import BaseScheduler

class ParallelActivation(BaseScheduler):
    """A scheduler which activates each agent once per step, in parallel, in random order,
    with the order reshuffled every step.

    This is equivalent to the NetLogo 'ask agents...' and is generally the
    default behavior for an ABM.

    Assumes that all agents have a step(model) method.
    """

    def __init__(self, model: Model, thread_count: Optional[int] = None) -> None:
        """If thread_count is not supplied, it will be the number of CPU cores on the system.
        """
        self.thread_count = thread_count
        self.pool = ProcessPoolExecutor(thread_count)
        super(ParallelActivation, self).__init__(model)

    def step_agent(agent):
        agent.step()
        
    def step(self) -> None:
        """Executes the step of all agents, in parallel, in
        random order.
        """
        agents = self.agent_buffer(shuffled=True)

        futures = [self.pool.submit(ParallelActivation.step_agent(agent), agent) for agent in agents] # sad and slow
        wait(futures)

        # futures = self.pool.map(ParallelActivation.step_agent, agents)
        # wait(list(futures))

        # with ProcessPoolExecutor(self.thread_count) as pool:
        #     list(pool.map(ParallelActivation.step_agent, agents))

        self.steps += 1
        self.time += 1

    def __del__(self):
        self.pool.shutdown()
