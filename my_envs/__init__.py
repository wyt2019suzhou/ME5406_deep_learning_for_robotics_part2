from gym.envs.registration import register

# Feeding
register(
    id='Feeding-v0',
    entry_point='my_envs.envs:Feeding',
    max_episode_steps=1000
)

# Feeding cooperation
register(
    id='FeedingCooperation-v0',
    entry_point='my_envs.envs:FeedingCop',
    max_episode_steps=1000
)



