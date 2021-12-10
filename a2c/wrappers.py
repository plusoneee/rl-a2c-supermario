import numpy as np
from skimage import transform
import gym
from gym.spaces import Box
from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        resize_obs = transform.resize(observation, self.shape)
        # cast float back to uint8
        resize_obs *= 255
        resize_obs = resize_obs.astype(np.uint8)
        return resize_obs

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

class CutomReward(gym.Wrapper):
    
    def __init__(self, env):
        super(CutomReward, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(1, 84, 84))
        self.curr_score = 0
        self.curr_x_pos = 0
        self.farthest_x_pos = 0
        self.not_forward_count_ = 0
    
    def stay_at_the_origin(self, info):
        return (info['x_pos'] < 30) and (info['time'] < 390)
    
    def did_not_go_forward(self, info):
        if (info['x_pos'] - self.curr_x_pos < 0):
            self.not_forward_count_ += 1
            return True
        return False

    def loginfo(self, reward, info):
        if info['life'] == 255:
            life = 'Game Over'
        else:
            life = f'Alive <{info["life"]}>'

        print(f"* Log@ {info['time']}, Status: {life}")
        print(f"* Reward: {reward/10.}\n* Coin: {info['coins']} \n* Score: {info['score']}\n* (X, Y):({info['x_pos']}, {info['y_pos']})")

    def step(self, action):
        state, reward, done, info = self.env.step(action)

        reward += (info["score"] - self.curr_score) / 10.
        self.curr_score = info["score"]
        
        if info['x_pos'] > self.farthest_x_pos:
            reward += 30
            self.farthest_x_pos = info['x_pos']

        if self.stay_at_the_origin(info):
            reward -= 20
    
        if self.did_not_go_forward(info):
            if self.not_forward_count_ == 4:
                reward -= 50
                self.not_forward_count_ = 0

        self.curr_x_pos = info['x_pos']
        if done:
            if info['time'] == 0:
                reward -= 50
                if info['x_pos'] < 200:
                    reward -= 100
                else:
                    reward -= 50

            if info["flag_get"]:
                reward += 100
            else:
                reward -= 30
            self.farthest_x_pos = 0

            reward += info['x_pos'] / (300 - info['time'])
            self.loginfo(reward/10, info)

        
        return state, reward / 10., done, info

    def reset(self):
        self.curr_score = 0
        return self.env.reset()

def apply_wrapper_env(env):
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env, keep_dim=False)
    env = ResizeObservation(env, shape=84)
    env = TransformObservation(env, f=lambda x: x / 255.)
    env = FrameStack(env, num_stack=4)
    env = CutomReward(env)

    return env