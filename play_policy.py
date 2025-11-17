import argparse
import time
import torch
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation, FlattenObservation
import ale_py
from gymnasium import error as gym_error

from dqnLL import DQNLL
from dqnALE import DQN


def load_policy_into_network(policy_path, net, device):
    """Try to load a state_dict from `policy_path` into `net`.
    The file may contain a plain state_dict or a dict with keys (e.g. 'q_network').
    """
    data = torch.load(policy_path, map_location=device)
    if isinstance(data, dict):
        for key in ("q_network", "model", "state_dict", "net"):
            if key in data:
                try:
                    net.load_state_dict(data[key])
                    return True
                except Exception:
                    pass
        try:
            net.load_state_dict(data)
            return True
        except Exception:
            try:
                net.load_state_dict(data, strict=False)
                return True
            except Exception:
                pass

            for v in data.values():
                if isinstance(v, dict):
                    try:
                        net.load_state_dict(v)
                        return True
                    except Exception:
                        try:
                            net.load_state_dict(v, strict=False)
                            return True
                        except Exception:
                            continue
                        
            def strip_prefix(d, prefix="module."):
                return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in d.items()}

            try:
                stripped = strip_prefix(data, "module.")
                net.load_state_dict(stripped)
                return True
            except Exception:
                try:
                    net.load_state_dict(stripped, strict=False)
                    return True
                except Exception:
                    pass

            def find_dicts(x):
                found = []
                if isinstance(x, dict):
                    found.append(x)
                    for v in x.values():
                        found.extend(find_dicts(v))
                return found

            for candidate in find_dicts(data):
                try:
                    net.load_state_dict(candidate)
                    return True
                except Exception:
                    try:
                        net.load_state_dict(candidate, strict=False)
                        return True
                    except Exception:
                        continue
            return False
    else:
        return False


def play_policy(policy_path: str, env_name: str = "ALE/Pong-v5",
                episodes: int = 5, render: bool = True):
    """Load a saved .pth policy and play `episodes` episodes on `env_name`.

    - `policy_path`: path to a .pth file containing the network state_dict
    - `env_name`: e.g. 'ALE/Pong-v5' or 'ALE/Breakout-v5'
    - `episodes`: number of episodes to play
    - `render`: start the env with render_mode='human' if True
    """
    device = "cpu"

    # Decide configuration based on env type (ALE vs others like LunarLander)
    is_ale = str(env_name).upper().startswith("ALE/")
    render_mode = "human" if render else None

    if is_ale:
        add_info = {'obs_type': "ram"}
        try:
            gym.register_envs(ale_py)
        except Exception:
            pass

        try:
            env = gym.make(env_name, render_mode=render_mode, **add_info)
        except gym_error.NamespaceNotFound as e:
            raise RuntimeError(
                "ALE environments are not available.\n"
                "Install dependencies: `pip install ale-py autorom gymnasium[atari]`\n"
                "Then install ROMs (AutoROM): run `python -m autorom` and accept licenses, or use `AutoROM().install()`.\n"
                "After that, retry. Original error: " + str(e)
            )

        env = FrameStackObservation(env, stack_size=4, padding_type="zero")
        env = FlattenObservation(env)
    else:
        add_info = {}
        try:
            env = gym.make(env_name, render_mode=render_mode, **add_info)
        except Exception as e:
            raise RuntimeError(f"Failed to create env '{env_name}': {e}")

        try:
            env = FlattenObservation(env)
        except Exception:
            pass
    
    obs_shape = env.observation_space.shape
    action_dim = env.action_space.n
    state_dim = int(np.prod(obs_shape))

    if is_ale: 
        net = DQN(state_dim, action_dim).to(device)
    else: 
        net = DQNLL(state_dim, action_dim).to(device)
    ok = load_policy_into_network(policy_path, net, device)
    if not ok:
        raise RuntimeError(f"Failed to load policy from {policy_path} into network")
    net.eval()

    rewards = []
    try:
        for ep in range(episodes):
            state, _ = env.reset()
            done = False
            total_reward = 0.0
            steps = 0
            while not done :
                s = torch.from_numpy(np.array(state)).float().to(device)
                if s.dim() == 1:
                    s = s.unsqueeze(0)
                with torch.no_grad():
                    q = net(s)
                    action = int(torch.argmax(q, dim=1).item())
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward += reward
                state = next_state
                steps += 1
                if render:
                    time.sleep(0.0)
            print(f"Episode {ep+1}/{episodes}: reward={total_reward:.2f} steps={steps}")
            rewards.append(total_reward)
    finally:
        env.close()

    avg = float(np.mean(rewards)) if rewards else 0.0
    print(f"Average reward over {len(rewards)} episodes: {avg:.2f}")
    return rewards


def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("policy", help=".pth policy file to load")
    p.add_argument("--env", default="ALE/Pong-v5", help="Gym env name")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--no-render", dest="render", action="store_false")
    args = p.parse_args()

    play_policy(args.policy, env_name=args.env, episodes=args.episodes, render=args.render)


if __name__ == "__main__":
    _cli()
