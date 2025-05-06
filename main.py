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

import numpy as np
import yaml
import os


def train_gail(config, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    # Initialize wandb
    

    best_loss = float('inf')
    os.makedirs(config['save_dir'], exist_ok=True)

    # Initialize components
    
    train_dataloader,val_dataloader, test_dataloader = create_dataloader(config['data_path'],batch_size=config['batch_size'])

    env = TennisEnvironment()
    policy = PolicyNetwork(state_dim=6, action_dim=2).to(device)
    discriminator = Discriminator(state_dim=6, action_dim=2).to(device)
    trpo = TRPO(policy, env, discriminator)
    lr = config["lr"]
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=lr)


    # Training loop
    disc_losses = []
    for epoch in range(config['epochs']):
        # --- Training Phase ---
        policy.train()
        discriminator.train()
        
        # Get training batch
        expert_batch = next(iter(train_dataloader))
        expert_states = expert_batch['states'].to(device)
        expert_actions = expert_batch['actions'].to(device)
        
        # Collect policy trajectories
        policy_trajectories = collect_policy_trajectories(policy, env, 
                                                        config['num_trajectories'], device)
        policy_states = torch.cat([t['states'] for t in policy_trajectories])
        policy_actions = torch.cat([t['actions'] for t in policy_trajectories])

        # Discriminator loss
        real_outputs = discriminator(expert_states, expert_actions)
        fake_outputs = discriminator(policy_states, policy_actions)
        disc_loss = -(torch.log(real_outputs + 1e-8).mean() + 
                    torch.log(1 - fake_outputs + 1e-8).mean())

        # Optimization step
        disc_optimizer.zero_grad()
        disc_loss.backward()
        disc_optimizer.step()

        # TRPO Update
        trpo_success = trpo.update_policy(policy_trajectories)

        # --- Validation Phase ---
        policy.eval()
        discriminator.eval()
        with torch.no_grad():
            # Get validation batch
            val_batch = next(iter(val_dataloader))
            val_states = val_batch['states'].to(device)
            val_actions = val_batch['actions'].to(device)
            
            # Collect validation trajectories
            val_policy_trajectories = collect_policy_trajectories(policy, env, 
                                                                config['num_trajectories'], device)
            val_policy_states = torch.cat([t['states'] for t in val_policy_trajectories])
            val_policy_actions = torch.cat([t['actions'] for t in val_policy_trajectories])

            # Validation outputs
            val_real = discriminator(val_states, val_actions)
            val_fake = discriminator(val_policy_states, val_policy_actions)
            val_disc_loss = -(torch.log(val_real + 1e-8).mean() + 
                            torch.log(1 - val_fake + 1e-8).mean())
            
            # Calculate validation metrics
            val_d_expert_acc = (val_real > 0.5).float().mean()
            val_d_policy_acc = (val_fake < 0.5).float().mean()
            val_d_total_acc = 0.5 * (val_d_expert_acc + val_d_policy_acc)

        # --- Logging ---
        # Training metrics
        train_d_expert_acc = (real_outputs > 0.5).float().mean()
        train_d_policy_acc = (fake_outputs < 0.5).float().mean()
        train_d_total_acc = 0.5 * (train_d_expert_acc + train_d_policy_acc)

        wandb.log({
            "epoch": epoch,
            "learning_rate": lr,
            "train/disc_loss": disc_loss.item(),
            "train/d_total_acc": train_d_total_acc.item(),
            "val/disc_loss": val_disc_loss.item(),
            "val/d_total_acc": val_d_total_acc.item(),
            "trpo_success": int(trpo_success)
        })

        # Print logging
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Train Loss: {disc_loss.item():.4f} | Val Loss: {val_disc_loss.item():.4f}")

        # --- Checkpointing ---
        if epoch % config['save_interval'] == 0:
            # Save checkpoint
            checkpoint_path = f"{config['save_dir']}/checkpoint_epoch{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'policy_state_dict': policy.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'train_loss': disc_loss.item(),
                'val_loss': val_disc_loss.item(),
            }, checkpoint_path)
            wandb.save(checkpoint_path)

        # Save best model based on validation loss
        if val_disc_loss.item() < best_loss:
            best_loss = val_disc_loss.item()
            best_model_path = f"{config['save_dir']}/best_model.pt"
            torch.save({
                'epoch': epoch,
                'policy_state_dict': policy.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'val_loss': val_disc_loss.item(),
            }, best_model_path)
            print(f"New best model at epoch {epoch} | Val Loss: {best_loss:.4f}")
            wandb.run.summary["best_val_loss"] = best_loss
            wandb.save(best_model_path)

        disc_losses.append(disc_loss.item())

    # --- Final Test Evaluation ---
    policy.eval()
    discriminator.eval()
    with torch.no_grad():
        test_batch = next(iter(test_dataloader))
        test_states = test_batch['states'].to(device)
        test_actions = test_batch['actions'].to(device)
        
        test_trajectories = collect_policy_trajectories(policy, env, 
                                                      config['num_trajectories'], device)
        test_policy_states = torch.cat([t['states'] for t in test_trajectories])
        test_policy_actions = torch.cat([t['actions'] for t in test_trajectories])

        test_real = discriminator(test_states, test_actions)
        test_fake = discriminator(test_policy_states, test_policy_actions)
        test_disc_loss = -(torch.log(test_real + 1e-8).mean() + 
                         torch.log(1 - test_fake + 1e-8).mean())
        
        test_d_total_acc = 0.5 * (
            (test_real > 0.5).float().mean() +
            (test_fake < 0.5).float().mean()
        )

    # Log final test results
    wandb.log({
        "test/disc_loss": test_disc_loss.item(),
        "test/d_total_acc": test_d_total_acc.item()
    })

    # Save final plots and cleanup
    plot_discriminator_loss(disc_losses, 
                           save_path=f"{config['save_dir']}/loss_curve.png")
    wandb.log({"loss_curve": wandb.Image(f"{config['save_dir']}/loss_curve.png")})
    wandb.finish()


def collect_policy_trajectories(policy, env, num_trajs, device):
    trajs = []
    policy = policy.to(device)
    
    for _ in range(num_trajs):
        state = env.reset()
        states = []
        actions = []
        done = False

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                action, _ = policy.act(state_tensor)
            next_state, _, done, _ = env.step(action.squeeze(0).cpu().numpy())

            states.append(state)
            actions.append(action.squeeze(0).cpu().numpy())
            state = next_state

        # Convert to tensors and move to device
        states_tensor = torch.FloatTensor(np.array(states)).to(device)
        actions_tensor = torch.FloatTensor(np.array(actions)).to(device)

        trajs.append({
            'states': states_tensor,
            'actions': actions_tensor
        })
    return trajs

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'training_config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config['training_config']


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config()
    
    # Access parameters
    print(f"Training for {config['epochs']} epochs")
    print(f"Data path: {config['data_path']}")

    wandb.login()
    run = wandb.init(
        project="tennis-gail",
        config=config,
        name=f"gail_e{config['epochs']}_t{config['num_trajectories']}_l{config['max_episode_length']}_lr_{config['lr']}"
    )
    train_gail(config, device)