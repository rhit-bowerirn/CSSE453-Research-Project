class Scout():
    def __init__(self, ID: int) -> None:
        self.ID = ID
        self.direction = 0

    def chooseDirection(self):
        print("Blah blah blah")

    def checkForGoal(self):
        #place for algorithm for checking the goal
        print("Scout #", self.ID, "Checking for goal")

    def signalMessageToNeighbor(self):
        #We can signal messages to neighbors like target found or danger... 
        #can be enums for certain messages
        print("Sending message to neighbor")    

    def listenForMessages(self):
        print("Listening for messages from our neighboring networker")
    
    def step(self):
        #set of functions a scout calls at each step
        print("Step")