# Simple SAC Implementation with pytorch

This project provides a simple implementation of the Soft Actor-Critic (SAC) algorithm for training an agent to control an environment from Gymnasium.

## Dependencies

- Python 3.8+
- PyTorch
- NumPy
- Gymnasium
- Matplotlib (for plotting)
- Pandas (for plotting)

You can install the necessary packages using:
```bash
pip install -r requirements.txt
```

# Usage
### Training: 
Train the SAC agent using the main.py script. You can customize training parameters using command-line arguments:
```bash
py main.py --epochs 20 --episodes_per_epoch 500 --lr 3e-4 --hidden_dims 256 --log_dir my_logs --ckpt_dir my_checkpoints --save_model
```
Available arguments:
```bash
--epochs: Number of training epochs.
--episodes_per_epoch: Number of episodes per epoch.
--lr: Learning rate.
--hidden_dims: Hidden dimensions for networks.
--buffer_size: Replay buffer size.
--learn_after: Steps before learning starts.
--batch_size: Batch size for updates.
--alpha: Temperature parameter alpha.
--final_alpha: Final value for temperature parameter alpha.
--anneal_alpha_steps: Number of steps to anneal alpha.
--env: Environment from gymnasium (default: "Pendulum-v1")
--log_dir: Directory to save logs (default: "Logs").
--ckpt_dir: Directory to save checkpoints (default: "Network").
--save_model: Save the trained model after training.
--force_cpu: Force using CPU even if CUDA is available.
--render: Changes render mode to "human".
```
## Plotting: 
Plot the training metrics (Average Q-values, Log Pi, Loss Pi, Loss Q) using the provided plot_logs.py script. Make sure to update the file_path variable in the script to point to your log file (e.g., my_logs/logs.txt if you used the --log_dir argument).
```bash
py logs/plot_logs.py
```

## Code Structure
```bash
SAC.py: Contains the implementation of the SAC agent.
agents.py: Defines the actor and critic networks.
replay_buffer.py: Implements the replay buffer for storing experiences.
logger.py: Handles logging of training metrics.
utils.py: Provides utility functions.
main.py: The main script for training the agent.
plot_logs.py: Script for plotting the training logs.
```

## Acknowledgements
This implementation is inspired by the SAC implementation from OpenAI Spinning Up. The code has been simplified and adapted for the Pendulum-v1 environment.