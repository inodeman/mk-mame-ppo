"""
Train a PPO agent on Mortal Kombat with parallel MAME environments.
"""

import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback
from mk_env import MortalKombatMAMEEnv
import os


def make_env(rank, seed=0):
    def _init():
        env = MortalKombatMAMEEnv()
        env.reset(seed=seed + rank)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-envs", type=int, default=6)
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--save-dir", type=str, default="models")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Parallel environments
    env = SubprocVecEnv([make_env(i) for i in range(args.n_envs)])
    # Optional: frame stack if not done inside env
    # env = VecFrameStack(env, n_stack=4)

    model = PPO(
        "CnnPolicy",
        env,
        learning_rate=2.5e-4,
        n_steps=512,
        batch_size=256,
        n_epochs=4,
        gamma=0.94,
        gae_lambda=0.95,
        clip_range=0.15,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log="logs/",
        device="auto",
    )

    checkpoint = CheckpointCallback(
        save_freq=50_000 // args.n_envs,
        save_path=args.save_dir,
        name_prefix="mk_ppo",
    )

    print(f"Training with {args.n_envs} parallel MAME environments...")
    model.learn(total_timesteps=args.timesteps, callback=checkpoint)
    model.save(os.path.join(args.save_dir, "mk_ppo_final"))
    print("Training finished.")


if __name__ == "__main__":
    main()
