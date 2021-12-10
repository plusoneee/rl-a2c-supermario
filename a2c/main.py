import torch
import gym_super_mario_bros
from wrappers import apply_wrapper_env
from mario import Mario

RENDER = False
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = apply_wrapper_env(env)
GAMMA = 0.99

# load agent
mario = Mario(
    n_actions=env.action_space.n,
    input_dim=(4, 84, 84),
    discount_factor=GAMMA
)

mario.model.train()
for i_episode in range(3000):
    print('=='*15)
    print('* Episode number :', i_episode)
    # observed space
    observation = env.reset()
    # add batch dim
    observation = torch.FloatTensor(observation).unsqueeze(0)
    done = False

    while not done:
        # Environment rendered after crossing return threshold
        if RENDER:
            env.render()

        # observation = observation.to(mario.device)
        value, action = mario.choose_action(observation)

        # load next state
        observation_, reward, done, info = env.step(action)
        observation_ = torch.FloatTensor(observation_).unsqueeze(0).to(mario.device)

        # save current state-action sequence
        mario.memory.push(observation, action, reward, value.max()) # greedy choice

        if done:
            
            obs = observation_
            
            # state-action value for new observation [V_(t+1)]
            Q_value, _ = mario.model(obs)
            
            # do not perform gradient update owing to operations on Q_value
            Q_value = Q_value.detach().to('cpu').numpy().max()  # greedy choice
            
            # update agent
            advantage = mario.update(Q_value, i_episode)

        # update
        observation = observation_

    if i_episode % 200 == 0:
        torch.save(mario.model.state_dict(), f'./chkpt/a2c-ep-{i_episode}.chkpt')
    