from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from dqn import DQN
import torch
import numpy as np
from copy import deepcopy
from evaluate import evaluate_HIV, evaluate_HIV_population
import os
from utils import load_config, ReplayBuffer
import logging

# Initialize logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
) 
class ProjectAgent:
    def __init__(self, config_path="src/config.yaml"):
        self.config = load_config(config_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.env = env
        self.nb_actions = self.env.action_space.n
        self.state_dim = self.env.observation_space.shape[0]

        dqn_config = self.config["dqn"]

        self.model = DQN(state_dim=self.state_dim, n_action=self.nb_actions, **dqn_config).to(self.device)
        self.target_model = deepcopy(self.model).to(self.device)

        train_config = self.config["training"]

        self.gamma = train_config.get("gamma", 0.95)
        self.batch_size = train_config.get("batch_size", 100)
        self.memory = ReplayBuffer(train_config["buffer_size"], self.device)
        self.epsilon_max = train_config.get("epsilon_max", 1.0)
        self.epsilon_min = train_config.get("epsilon_min", 0.01)
        self.epsilon_stop = train_config.get("epsilon_decay_period", 1000)
        self.epsilon_delay = train_config.get("epsilon_delay_decay", 100)
        self.epsilon_step = (self.epsilon_max - self.epsilon_min) / self.epsilon_stop
        self.criterion = getattr(torch.nn, train_config.get("criterion", "MSELoss"))()
        lr = train_config.get("learning_rate", 0.001)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.nb_gradient_steps = train_config.get("gradient_steps", 1)
        self.update_target_strategy = train_config.get("update_target_strategy", "replace")
        self.update_target_freq = train_config.get("update_target_freq", 50)
        self.update_target_tau = train_config.get("update_target_tau", 0.005)
        self.path = os.path.join(os.getcwd(), "models")


    def act(self, observation, use_random=False):
        if use_random:
            return np.random.randint(self.nb_actions)
        else:
            with torch.no_grad():
                Q = self.model(torch.Tensor(observation).unsqueeze(0).to(self.device))
                return torch.argmax(Q).item()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self):
        self.model.load_state_dict(torch.load("src/model_all_10.pth", map_location=self.device))
        self.model.eval()

    def greedy_action(self, network, state):
        with torch.no_grad():
            Q = network(torch.Tensor(state).unsqueeze(0).to(self.device))
            return torch.argmax(Q).item()

    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.target_model(Y).max(1)[0].detach()
            update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 
    
    def train(self, max_episode=250):
        """
        Trains the agent using Deep Q-Learning.
        
        Args:
            max_episode (int): Maximum number of episodes for training.
        
        Returns:
            list: Cumulative rewards for each episode.
        """
        epsilon = self.epsilon_max
        step = 0
        episode_rewards = []
        prev_val_score = 0

        for episode in range(1, max_episode + 1):
            episode_cum_reward, step, epsilon = self.run_episode(epsilon, step)

            # Validate and update best model
            # val_score = self.validate_agent(episode, prev_val_score)
            if episode_cum_reward > prev_val_score:
                prev_val_score = episode_cum_reward
                self.save_best_model(episode, episode_cum_reward)

            episode_rewards.append(episode_cum_reward)

            # Log progress
            logger.info(f"Episode {episode:3d} | Epsilon: {epsilon:.2f} | "
                        f"Memory Size: {len(self.memory):5d} | "
                        f"Return: {episode_cum_reward:.2E} | ")
                        # f"Validation Score: {val_score:.2E}")

        # Save the final model
        self.finalize_training(prev_val_score)
        return episode_rewards

    def reset_environment(self):
        """Resets the environment and returns the initial state."""
        return self.env.reset()

    def update_epsilon(self, epsilon, step):
        """Updates the exploration rate (epsilon) based on the current step."""
        if step > self.epsilon_delay:
            epsilon = max(self.epsilon_min, epsilon - self.epsilon_step)
        return epsilon

    def run_episode(self, epsilon, step):
        """Executes a single episode and returns the cumulative reward."""
        episode_cum_reward = 0
        state, _ = self.reset_environment()
        done, trunc = False, False

        while not done and not trunc:
            epsilon = self.update_epsilon(epsilon, step)
            action = self.select_action(state, epsilon)
            next_state, reward, done, trunc, _ = self.env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward

            # Perform gradient steps
            for _ in range(self.nb_gradient_steps):
                self.gradient_step()

            # Update target model periodically
            if step % self.update_target_freq == 0:
                self.target_model.load_state_dict(self.model.state_dict())
            
            step += 1
            state = next_state

        return episode_cum_reward, step, epsilon

    def select_action(self, state, epsilon):
        """Selects an action using epsilon-greedy policy."""
        if np.random.rand() < epsilon:
            return self.env.action_space.sample()
        return self.greedy_action(self.model, state)

    def validate_agent(self, episode, prev_val_score):
        """Validates the agent and returns the validation score."""
        if episode % 10 == 0:  # Validate every 10 episodes
            return evaluate_HIV(agent=self, nb_episode=1)
        return prev_val_score

    def save_best_model(self, episode, val_score):
        """Saves the current best model."""
        model_path = os.path.join(self.path, f"model_episode_{episode}_score:{val_score:.2E}.pth")
        self.best_model = deepcopy(self.model).to(self.device)
        self.save(model_path)
        logger.info(f"Best model saved with score {val_score:.2f}")

    def finalize_training(self, val_score):
        """Saves the final model after training."""
        final_path = os.path.join(self.path, f"model_episode_final_score:{val_score:.2E}.pth")
        self.model.load_state_dict(self.best_model.state_dict())
        self.save(final_path)
        logger.info(f"Final model saved with score {val_score:.2f}")
    
# if __name__ == "__main__":
#     env = TimeLimit(env=HIVPatient(domain_randomization=True), max_episode_steps=200)
#     agent = ProjectAgent(env=env)
    
#     rewards = agent.train(max_episode=250)
#     print(rewards)

#     score_agent: float = evaluate_HIV(agent=agent, nb_episode=1)
#     score_agent_dr: float = evaluate_HIV_population(agent=agent, nb_episode=15)

#     with open(file="score.txt", mode="w") as f:
#         f.write(f"{score_agent}\n{score_agent_dr}")