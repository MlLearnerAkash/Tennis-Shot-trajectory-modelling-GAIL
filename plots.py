#### Utility:
import matplotlib.pyplot as plt
import numpy as np

def plot_discriminator_loss(loss_history, window_size=10, save_path=None):
    """
    Plot discriminator loss with smoothing and training analysis

    Args:
        loss_history (list): List of discriminator loss values
        window_size (int): Moving average window size for smoothing
        save_path (str): Optional path to save the plot
    """
    plt.figure(figsize=(12, 6))

    # Raw loss values
    epochs = np.arange(len(loss_history))
    plt.plot(epochs, loss_history, 'b-', alpha=0.3, label='Raw Loss')

    # Smoothed loss
    if len(loss_history) >= window_size:
        weights = np.repeat(1.0, window_size)/window_size
        smoothed = np.convolve(loss_history, weights, mode='valid')
        plt.plot(epochs[window_size-1:], smoothed, 'r-', lw=2,
                label=f'Smoothed ({window_size} epoch MA)')

    # Formatting
    plt.title("Discriminator Loss Evolution", fontsize=14)
    plt.xlabel("Training Epochs", fontsize=12)
    plt.ylabel("Loss Value", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Add training phase annotations
    if len(loss_history) > 0:
        final_loss = loss_history[-1]
        min_loss = np.min(loss_history)
        plt.annotate(f'Final Loss: {final_loss:.4f}',
                    xy=(0.98, 0.95), xycoords='axes fraction',
                    ha='right', va='top', color='darkred')

        plt.annotate(f'Minimum Loss: {min_loss:.4f}',
                    xy=(0.98, 0.88), xycoords='axes fraction',
                    ha='right', va='top', color='darkgreen')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    