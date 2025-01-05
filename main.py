import argparse
from SAC import SAC
from logger import Logger 
import utils
import gymnasium as gym
import torch

def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    print(f"Using device: {device}")

    render_mode = "human" if args.render else "rgb_array"

    env = gym.make(args.env, render_mode=render_mode)  # Use args.env

    agent = SAC(env.observation_space.shape[0], env.action_space.shape[0], env.action_space.high[0],
                 hidden_dims=args.hidden_dims, lr=args.lr,
                 buffer_size=args.buffer_size, learn_after=args.learn_after, alpha=args.alpha,
                 ckpt_dir=args.ckpt_dir, device=device) # Pass device to the agent
    
    logger = Logger(output_dir=args.log_dir)
    agent.logger = logger

    observation, info = env.reset()

    for epoch in range(args.epochs):
        for episode in range(args.episodes_per_epoch):
            episode_reward = 0
            episode_over = False
            while not episode_over:
                if not args.const_alpha:  # use contant value for alpha if flag set
                    agent.alpha = utils.get_alpha(agent.current_step, 0.8, args.final_alpha, args.anneal_alpha_steps)

                action = agent.get_action(observation)

                next_observation, reward, terminated, truncated, info = env.step(action)

                agent.replay_buffer.add(observation, action, next_observation, reward, terminated or truncated)

                if agent.replay_buffer.size > args.learn_after:
                    agent.update(args.batch_size)

                observation = next_observation
                episode_reward += reward
                episode_over = terminated or truncated

        print(f"Epoch: {epoch+1}/{args.epochs}, Reward: {episode_reward:.2f}, Total steps: {agent.current_step}")

        if agent.current_step > agent.learn_after:
            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('TotalEnvInteracts', agent.current_step)
            logger.log_tabular('Q1Vals', with_min_and_max=True)
            logger.log_tabular('Q2Vals', with_min_and_max=True)
            logger.log_tabular('LogPi', with_min_and_max=True)
            logger.log_tabular('LossPi', avg_only=True)
            logger.log_tabular('LossQ', avg_only=True)
            logger.dump_tabular()

    if agent.current_step < agent.learn_after:
        print("Total ammount of steps was lower than learn_after resulting in no training beeing done.")

    if args.save_model:  # Save model if flag is set
        agent.save(path=args.ckpt_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SAC agent.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--episodes_per_epoch", type=int, default=1_000, help="Number of episodes per epoch.")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument("--hidden_dims", type=int, default=64, help="Hidden dimensions for networks.")
    parser.add_argument("--buffer_size", type=int, default=int(1e6), help="Replay buffer size.")
    parser.add_argument("--learn_after", type=int, default=1_000, help="Steps before learning starts.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for updates.")
    parser.add_argument("--alpha", type=float, default=0.8, help="Temperature parameter alpha.")
    parser.add_argument("--const_alpha", action="store_true", help="Use constant value for alpha.")
    parser.add_argument("--final_alpha", type=float, default=0.1, help="Final value for temperature parameter alpha.")
    parser.add_argument("--anneal_alpha_steps", type=int, default=100_000, help="Number of steps to anneal alpha.")
    parser.add_argument("--log_dir", type=str, default="Logs", help="Directory to save logs.")
    parser.add_argument("--ckpt_dir", type=str, default="Network", help="Directory to save checkpoints.")
    parser.add_argument("--save_model", action="store_true", help="Save trained model.")
    parser.add_argument("--force_cpu", action="store_true", help="Force using CPU even if CUDA is available.")
    parser.add_argument("--env", type=str, default="Pendulum-v1", help="Gym environment ID.")
    parser.add_argument("--render", action="store_true", help="Render the environment.")

    args = parser.parse_args()
    main(args)
