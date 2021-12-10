import torch
import gym_super_mario_bros
from wrappers import apply_wrapper_env
from mario import Mario

RENDER = False
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = apply_wrapper_env(env)
# hyper parameters
GAMMA = 0.99

# load agent
mario = Mario(
    n_actions=env.action_space.n,
    input_dim=(4, 84, 84),
    discount_factor=GAMMA
)

for i_episode in range(3000):
    # observed space
    observation = env.reset()
    # add batch dim
    observation = torch.FloatTensor(observation).unsqueeze(0)

    while True:
        # Environment rendered after crossing return threshold
        if RENDER:
            env.render()

        # observation = observation.to(mario.device)
        value, action = mario.choose_action(observation)
        done = False

        # load next state
        observation_, reward, done, info = env.step(action)
        observation_ = torch.FloatTensor(observation_).unsqueeze(0).to(mario.device)

        # save current state-action sequence
        mario.memory.push(observation, action, reward, value.max()) # greedy choice

        if done:
            env.reset()
            # convert new observation to tensor
            obs = observation_
            
            # state-action value for new observation [V_(t+1)]
            Q_value, _ = mario.model(obs)
            
            # do not perform gradient update owing to operations on Q_value
            Q_value = Q_value.detach().to('cpu').numpy().max()  # greedy choice
            
            # update agent
            advantage = mario.update(Q_value, i_episode)
        
        # update
        observation = observation_