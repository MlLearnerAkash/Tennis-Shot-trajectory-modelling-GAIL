import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict
import os
import torch
import wandb
from dataset.dataloader import create_dataloader
from utils.env import TennisEnvironment
from model import PolicyNetwork, Discriminator
from src.trpo import TRPO
from plots import plot_discriminator_loss
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from dataset.data_prep import TennisTrajectoryDataset

import numpy as np
import yaml
from utils.constants import *
import os

def load_inference_components(model_path: str, data_path: str) -> Tuple:
    """Load trained model and dataset scalers for inference"""
    # Initialize components
    env = TennisEnvironment()
    dataset = TennisTrajectoryDataset(config['data_path'])
    
    # Create models
    policy = PolicyNetwork(state_dim=6, action_dim=2)
    discriminator = Discriminator(state_dim=6, action_dim=2)

    # Load checkpoint
    checkpoint = torch.load(os.path.join(model_path, "best_model.pt"))
    policy.load_state_dict(checkpoint['policy_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    
    # Set to evaluation mode
    policy.eval()
    discriminator.eval()
    
    return env, policy, dataset.scalers

def initialize_environment_from_data(env: TennisEnvironment,
                                   scalers: Dict,
                                   sample_idx: int = 10) -> np.ndarray:
    """Initialize environment state from dataset sample"""
    # Get sample from dataset
    dataset = TennisTrajectoryDataset(config['data_path'])
    sample = dataset[sample_idx]
    initial_state_normalized = sample['states'][0].numpy()

    # Denormalization function
    def denormalize(col: str, value: float) -> float:
        return value * scalers[col][1] + scalers[col][0]

    # Set environment state
    env.ball_position = np.array([
        denormalize('ball_x', initial_state_normalized[0]),
        denormalize('ball_y', initial_state_normalized[1])
    ], dtype=np.float32)
    
    env.ball_velocity = np.array([
        denormalize('velocity_x', initial_state_normalized[2]),
        denormalize('velocity_y', initial_state_normalized[3])
    ], dtype=np.float32)
    
    env.distance_since_last_hit = denormalize('distance_travelled', initial_state_normalized[4])
    env.direction = 1 if env.ball_velocity[0] > 0 else 0
    
    return env._get_state()

def generate_trajectory(env: TennisEnvironment,
                       policy: PolicyNetwork,
                       scalers: Dict,
                       max_steps: int = 10000) -> Tuple:
    """Generate trajectory using trained policy"""
    state = env._get_state()
    actual_positions = [env.ball_position.copy()]
    predicted_positions = []

    # Normalization function
    def normalize_state(s: np.ndarray) -> np.ndarray:
        return np.array([
            (s[0] - scalers['ball_x'][0]) / scalers['ball_x'][1],
            (s[1] - scalers['ball_y'][0]) / scalers['ball_y'][1],
            (s[2] - scalers['velocity_x'][0]) / scalers['velocity_x'][1],
            (s[3] - scalers['velocity_y'][0]) / scalers['velocity_y'][1],
            (s[4] - scalers['distance_travelled'][0]) / scalers['distance_travelled'][1],
            s[5]
        ], dtype=np.float32)

    done = False
    while not done and len(actual_positions) < max_steps:
        # Normalize and predict
        norm_state = normalize_state(state)
        with torch.no_grad():
            action = policy(torch.FloatTensor(norm_state).unsqueeze(0))[0].squeeze().numpy()
        
        # Store prediction
        current_pos = actual_positions[-1]
        predicted_pos = current_pos + action * scalers['velocity_x'][1]
        predicted_positions.append(predicted_pos)
        
        # Environment step
        next_state, _, done, _ = env.step(action)
        actual_positions.append(next_state[:2])
        state = next_state

    return np.array(actual_positions), np.array(predicted_positions)

def plot_trajectory(actual: np.ndarray,
                   predicted: np.ndarray,
                   save_path: str = None) -> None:
    """Visualize generated trajectory with court layout"""
    plt.figure(figsize=(12, 8))
    
    # Court dimensions
    court_length, court_width = COURT_DIMENSIONS[0], COURT_DIMENSIONS[1]
    
    # Draw court
    plt.plot([0, court_length], [0, 0], 'k-', lw=3, label='Baseline')
    plt.plot([0, court_length], [court_width, court_width], 'k--', lw=2, label='Net')
    plt.plot([court_length/2, court_length/2], [0, court_width], 'k:')
    
    # Plot trajectories
    plt.plot(actual[::1000, 0], actual[::1000, 1], 'g-o', label='Actual Path')
    plt.plot(predicted[::1000, 0], predicted[::1000, 1], 'b--s', label='Predicted Path')
    
    # Error vectors
    for i in range(0, len(predicted), 1000):
        plt.annotate('', xy=predicted[i], xytext=actual[i],
                    arrowprops=dict(arrowstyle="->", color='r', alpha=0.5))
    
    # Formatting
    plt.title("Ball Trajectory Prediction", fontsize=14)
    plt.xlabel("X Position (pixels)", fontsize=12)
    plt.ylabel("Y Position (pixels)", fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.xlim(-100, court_length+100)
    plt.ylim(-100, court_width+100)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def run_inference(model_path: str, data_path: str) -> None:
    """End-to-end inference pipeline"""
    # Load components
    env, policy, scalers = load_inference_components(model_path, data_path)
    
    # Initialize environment
    initialize_environment_from_data(env, scalers)
    
    # Generate trajectory
    actual, predicted = generate_trajectory(env, policy, scalers)
    
    # Visualize results
    plot_trajectory(actual, predicted, "trajectory_prediction.png")
    
    # Print metrics
    final_error = np.linalg.norm(actual[-1] - predicted[-1])
    print(f"Final Position Error: {final_error:.2f} pixels")
    print(f"Trajectory Length: {len(actual)} steps")

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'training_config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config['training_config']

# Example usage
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config()
    
    # Access parameters
 

    run_inference(config['save_dir'], config['data_path'])