"""RL training utilities.

Provides training pipelines for execution and sizing RL agents.
"""

from pathlib import Path
from typing import Any
import numpy as np

from src.utils.logging import get_logger


logger = get_logger(__name__)


class RLTrainer:
    """RL training pipeline for trading agents.

    Supports:
    - Behavior cloning (imitation learning)
    - Offline RL (CQL, IQL)
    - Online RL (PPO)
    """

    def __init__(
        self,
        env_name: str = "execution",
        algorithm: str = "ppo",
        output_dir: Path | None = None,
    ):
        """Initialize trainer.

        Args:
            env_name: Environment name ('execution' or 'sizing').
            algorithm: RL algorithm ('ppo', 'cql', 'iql', 'bc').
            output_dir: Directory for checkpoints and logs.
        """
        self.env_name = env_name
        self.algorithm = algorithm.lower()
        self.output_dir = output_dir or Path("./models/rl")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.env = None
        self.model = None

    def create_env(self, **kwargs) -> Any:
        """Create the training environment.

        Args:
            **kwargs: Environment configuration.

        Returns:
            Gym environment instance.
        """
        if self.env_name == "execution":
            from src.rl.envs.execution_env import ExecutionEnv
            self.env = ExecutionEnv(**kwargs)
        elif self.env_name == "sizing":
            from src.rl.envs.sizing_env import SizingEnv
            self.env = SizingEnv(**kwargs)
        else:
            raise ValueError(f"Unknown environment: {self.env_name}")

        return self.env

    def train_ppo(
        self,
        total_timesteps: int = 100000,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        **kwargs,
    ) -> dict:
        """Train using PPO algorithm.

        Args:
            total_timesteps: Total environment steps for training.
            learning_rate: Learning rate.
            n_steps: Steps per rollout.
            batch_size: Minibatch size.
            n_epochs: Epochs per update.
            **kwargs: Additional PPO parameters.

        Returns:
            Training metrics.
        """
        try:
            from stable_baselines3 import PPO
            from stable_baselines3.common.callbacks import EvalCallback
        except ImportError:
            logger.error("stable-baselines3 required for PPO training")
            raise

        if self.env is None:
            self.create_env()

        logger.info(f"Starting PPO training for {total_timesteps} steps")

        # Create model
        self.model = PPO(
            "MlpPolicy",
            self.env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            verbose=1,
            **kwargs,
        )

        # Create eval callback
        eval_callback = EvalCallback(
            self.env,
            best_model_save_path=str(self.output_dir),
            log_path=str(self.output_dir / "logs"),
            eval_freq=5000,
            deterministic=True,
        )

        # Train
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
        )

        # Save final model
        model_path = self.output_dir / f"ppo_{self.env_name}_final.zip"
        self.model.save(str(model_path))
        logger.info(f"Saved model to {model_path}")

        return {"model_path": str(model_path)}

    def train_offline(
        self,
        dataset: dict,
        n_steps: int = 100000,
        algorithm: str = "cql",
        **kwargs,
    ) -> dict:
        """Train using offline RL (CQL or IQL).

        Args:
            dataset: Dataset dict with 'observations', 'actions', 'rewards', etc.
            n_steps: Training steps.
            algorithm: 'cql' or 'iql'.
            **kwargs: Additional algorithm parameters.

        Returns:
            Training metrics.
        """
        try:
            import d3rlpy
        except ImportError:
            logger.error("d3rlpy required for offline RL training")
            raise

        logger.info(f"Starting {algorithm.upper()} offline training")

        # Create d3rlpy dataset
        d3_dataset = d3rlpy.dataset.MDPDataset(
            observations=dataset["observations"],
            actions=dataset["actions"],
            rewards=dataset["rewards"],
            terminals=dataset["terminals"],
        )

        # Create algorithm
        if algorithm == "cql":
            self.model = d3rlpy.algos.CQL()
        elif algorithm == "iql":
            self.model = d3rlpy.algos.IQL()
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        # Train
        self.model.fit(
            d3_dataset,
            n_steps=n_steps,
            logdir=str(self.output_dir / "logs"),
        )

        # Save
        model_path = self.output_dir / f"{algorithm}_{self.env_name}.d3"
        self.model.save(str(model_path))
        logger.info(f"Saved model to {model_path}")

        return {"model_path": str(model_path)}

    def behavior_cloning(
        self,
        expert_data: dict,
        n_epochs: int = 100,
        batch_size: int = 64,
        **kwargs,
    ) -> dict:
        """Train via behavior cloning (imitation learning).

        Args:
            expert_data: Dict with 'observations' and 'actions'.
            n_epochs: Training epochs.
            batch_size: Batch size.
            **kwargs: Additional parameters.

        Returns:
            Training metrics.
        """
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError:
            logger.error("PyTorch required for behavior cloning")
            raise

        logger.info("Starting behavior cloning")

        observations = torch.FloatTensor(expert_data["observations"])
        actions = torch.FloatTensor(expert_data["actions"])

        dataset = TensorDataset(observations, actions)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Simple MLP policy
        obs_dim = observations.shape[1]
        act_dim = actions.shape[1] if len(actions.shape) > 1 else 1

        policy = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim),
            nn.Tanh(),
        )

        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        losses = []
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            for obs, act in loader:
                optimizer.zero_grad()
                pred = policy(obs)
                if len(act.shape) == 1:
                    act = act.unsqueeze(1)
                loss = criterion(pred, act)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)
            losses.append(avg_loss)

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")

        # Save
        model_path = self.output_dir / f"bc_{self.env_name}.pt"
        torch.save(policy.state_dict(), model_path)
        logger.info(f"Saved BC model to {model_path}")

        return {
            "model_path": str(model_path),
            "final_loss": losses[-1],
        }

    def evaluate(self, n_episodes: int = 100) -> dict:
        """Evaluate trained model.

        Args:
            n_episodes: Number of evaluation episodes.

        Returns:
            Evaluation metrics.
        """
        if self.model is None:
            raise ValueError("No model loaded")

        if self.env is None:
            self.create_env()

        rewards = []
        episode_lengths = []

        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            done = False
            episode_reward = 0
            length = 0

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                episode_reward += reward
                length += 1
                done = terminated or truncated

            rewards.append(episode_reward)
            episode_lengths.append(length)

        return {
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "mean_length": np.mean(episode_lengths),
            "n_episodes": n_episodes,
        }
