from stable_baselines3 import PPO
from tmai.env.TMNFEnv import TrackmaniaEnv
import time
import sys

if __name__ == "__main__":
    env = TrackmaniaEnv(action_space="gamepad")
    if (input("pretrain? (y/n)") == "y"):
        model_name = input("model name: ")
        model = PPO.load("ppo_models/" + model_name, env=env, verbose=1, n_steps=128, batch_size=3072, learning_rate=0.0001, device="cpu", tensorboard_log="./tensorboard/")
        print("model loaded")
    else:
        model = PPO("MlpPolicy", env, verbose=1, n_steps=128, batch_size=3072, learning_rate=0.0001, device="cpu", tensorboard_log="./tensorboard/")
        print("model created")
    
    while True:
        time_step = int(input("Enter time step: "))
        print("training for", time_step, "time steps")
        time.sleep(5)
        env.reset()
        model = model.learn(total_timesteps=time_step, progress_bar=True, log_interval=1, tb_log_name="ppo_model", reset_num_timesteps=False)
        model.save("ppo_models/ppo_model_" + str(model.num_timesteps))
        print("model saved")
        env.reset()
        if (input("continue? (y/n)") == "n"):
            break
    
    print("model closed")
    sys.exit(0)