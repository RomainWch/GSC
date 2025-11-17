import argparse
import gymnasium as gym
import numpy as np

from dqnALE import DQNAgent

import matplotlib.pyplot as plt
import torch
import os
import re
import ale_py
from gymnasium.wrappers import FrameStackObservation, FlattenObservation

# ------------------ CLI and Environment Setup ------------------
parser = argparse.ArgumentParser()
parser.add_argument("--env", default='ALE/Pong-v5', help="Gym env name to use")
parser.add_argument("--episodes", type=int, default=1501, help="Number of episodes to train")
args = parser.parse_args()

env_name = args.env
episodes = args.episodes

add_info = {'obs_type': "ram"}
gym.register_envs(ale_py)

env = gym.make(env_name, **add_info)
env = FrameStackObservation(env, stack_size=4, padding_type="zero")
env = FlattenObservation(env)

# --------- Saving Policy ---------
save_policy = True       # Enable/Disable saving
save_every = 300        # Save every X episodes
PARAM = "test_numero_3_"  # Description of the parameters

env_base = env_name.split('/')[-1]
env_base = re.sub(r"-v\d+$", "", env_base)
safe_env_base = env_base.replace('/', '_').replace(':', '_')
save_path = f"policies/{safe_env_base}_" + PARAM

# ------------------ Agent Setup ------------------

agent = DQNAgent(
    env,
    gamma=0.99,
    alpha=0.0001,
    beta=0.4,
    epsilon=1.0,
    epsilon_decay_steps=2000000,
    min_epsilon=0.01,
    target_update_freq=5000,
    buffer_size=500000,
    batch_size=64,
)

rewards = []

# ---------------- Main Training Loop ------------------
avg_rewards = 0
number_episodes = 0

for episode in range(episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done and total_reward < 500:
        action = agent.play(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.update(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
    
    avg_rewards += 1/ (number_episodes + 1) * (total_reward - avg_rewards)
    if number_episodes > 100:

        avg_rewards = 0
        number_episodes = 0
    number_episodes += 1

    print(f"Episode {episode} | Steps: {agent.steps_done} | Avg Reward: {avg_rewards:.2f} | Epsilon: {agent.epsilon:.3f} | Total Reward: {total_reward:.2f}")
    rewards.append(total_reward)

    # ---- Save policy periodically ----
    save_path_2 = save_path + f"avg{int(avg_rewards)}_ep{episode}.pth"
    if save_policy and episode % save_every == 0 and episode > 0:
        # ensure the parent directory exists before saving to avoid RuntimeError
        save_dir = os.path.dirname(save_path_2)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        if hasattr(agent, "q_network"):
            torch.save(agent.q_network.state_dict(), save_path_2)
            print(f"✅ Policy (q_network) saved at episode {episode} -> {save_path_2}")
        elif hasattr(agent, "model"):
            torch.save(agent.model.state_dict(), save_path_2)
            print(f"✅ Policy (model) saved at episode {episode} -> {save_path_2}")
        else:
            print("⚠️ No neural network found in agent, skipping save...")

    env.close()

# Plot average reward over last 100 episodes
avg_rewards = [np.mean(rewards[max(0, i - 100): i + 1]) for i in range(len(rewards))]
plt.plot(avg_rewards)
plt.xlabel("Episode")
plt.ylabel("Average Reward (100 ep)")
plt.title(f"Monte Carlo on {env_name}")
plt.savefig("rewardsDQN/rewards.png", dpi=300, bbox_inches="tight")
plt.show()
