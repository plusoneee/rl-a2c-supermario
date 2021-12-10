class Memory:
    
    def __init__(self):
        self.ep_obs = []
        self.ep_as = []
        self.ep_rewards = []
        self.ep_values = []

    def clear(self):
        self.ep_obs = []
        self.ep_as = []
        self.ep_rewards = []
        self.ep_values = []

    def push(self, observation, action, reward, value):
        self.ep_obs.append(observation)
        self.ep_as.append(action)
        self.ep_rewards.append(reward)
        self.ep_values.append(value)
