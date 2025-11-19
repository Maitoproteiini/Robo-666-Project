import safety_gymnasium
env = safety_gymnasium.make('SafetyCarGoal1-v0', render_mode='human')

obs, info = env.reset()
# Set seeds
# obs, _ = env.reset(seed=0)
terminated, truncated = False, False
ep_ret, ep_cost = 0, 0

assert env.observation_space.contains(obs)
act = env.action_space.sample()
assert env.action_space.contains(act)
# modified for Safe RL, added cost
obs, reward, cost, terminated, truncated, info = env.step(act)
print(f"Type: {type(obs)}")
print(f"Shape: {obs.shape}")
print(f"Observation: {obs}")
print(f"Env observation: {env.observation_space}")

env.close()