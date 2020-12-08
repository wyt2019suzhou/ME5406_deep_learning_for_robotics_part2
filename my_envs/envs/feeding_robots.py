from my_envs.envs.feeding import FeedingEnv

class Feeding(FeedingEnv):
    def __init__(self):
        super(Feeding, self).__init__(human_control=False)

class FeedingCop(FeedingEnv):
    def __init__(self):
        super(FeedingCop, self).__init__(human_control=True)


