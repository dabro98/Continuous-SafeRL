import sys
import os
import json
import time
import numpy as np
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam
# add boxenv folder to path - resloves import problem
sys.path.insert(0, os.getcwd() + '/boxenvmain')
from boxdynamics import BoxEnv
from torch.utils.tensorboard import SummaryWriter

CONFIG_PATH = sys.argv[1]

def loadBoxEnv(env_path):
    env = BoxEnv()
    env.world_and_load_design_without_user_input(env_path)
    return env

def loadConfig(filename=CONFIG_PATH):
    with open(filename, "r") as file:
        params = json.load(file)
    return params
    
class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """
    def __init__(self, critical_area_reward, log_std_init,
                n_steps, n_epochs, env_name, seed, verbose=1):
        self.critical_area_reward = critical_area_reward
        self.log_std_init = log_std_init
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.env_name = env_name
        self.seed = str(seed)
        self.n_violations = 0
        super().__init__(verbose)

    def get_violation(self):
        infos = self.locals["infos"]
        violated = infos[0]["violation"]
        return violated

    def _on_training_start(self) -> None:
            hparam_dict = {
                "algorithm": self.model.__class__.__name__,
                "critical_area_reward": self.critical_area_reward,
                "log_std_init": self.log_std_init,
                "n_steps": self.n_steps,
                "n_epochs": self.n_epochs,
                "env_name": self.env_name,
                "seed": self.seed
            }
            # define the metrics that will appear in the `HPARAMS` Tensorboard tab by referencing their tag
            # Tensorbaord will find & display metrics from the `SCALARS` tab
            metric_dict = {
                "rollout/ep_len_mean": 0,
                "rollout/ep_rew_mean": 0,
                "train/value_loss": 0.0,
                "rollout/n_violations": 0,
            }
            self.logger.record(
                "hparams",
                HParam(hparam_dict, metric_dict),
                exclude=("stdout", "log", "json", "csv"),
            )
    
    def _on_step(self) -> bool:
        # Log number of violations
        if self.get_violation():
            self.n_violations +=1
            
        self.logger.record("rollout/n_violations", self.n_violations)
        return True
    
def main():
    params = loadConfig()

    env = loadBoxEnv(params["env_path"])

    env.set_reset_n_steps(params["n_steps"])

    curr_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    seed = int(time.time())
    env_name = params["env_path"].split("/")[-1].split(".")[0]
    if env_name.find("_"):
        env_name = env_name.split("_")[0]
    adv_log_name = f"./logs/{params['algorithm']}/{env_name}"

    policy_kwargs = dict(log_std_init=params["log_std_init"])
    
    callback = TensorboardCallback(env.get_critical_area_reward(), params["log_std_init"], 
                                    params["n_steps"], params["n_epochs"], env_name, seed)

    for x in range(1, 11):
        adv_model_name = f"./models/{params['algorithm']}/{env_name}/{curr_time}_{x}"
        adv_eval_name = adv_model_name.replace("models", "eval")

        model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1, 
                    n_steps=params["n_steps"], n_epochs=params["n_epochs"],
                    tensorboard_log=adv_log_name, seed=seed)
        
        
        print("start learning iteration: {x}")
        model.learn(total_timesteps=params["total_steps"], progress_bar=True, 
                    tb_log_name=curr_time, 
                    callback=callback)
        
        model.save(adv_model_name)

        # To check out the model - pass model name here and comment out the (previous) reset, except 84,86
        #adv_model_name = "models/PPO/t1/2023-09-27_19:13:29.zip"
        model = PPO.load(adv_model_name)

        ########## EVALUATION ##########
        print(f'model {adv_model_name} loaded - start evaluation')

        writer = SummaryWriter(log_dir=adv_eval_name)
        n_violations = 0
        n_reached_goal = 0
        rewards = []

        obs = env.reset()
        env.set_reset_n_steps(np.inf)


        steps = params["eval_steps"]
        for step in range(0, steps):
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)

            rewards.append(reward)
            #writer.add_scalar("Eval/curr_reward", np.mean(rewards), step)

            if info["violation"]:
                n_violations+=1
            if done:
                n_reached_goal+=1
                obs = env.reset()
                continue

        mean_reward = np.mean(rewards)

        writer.add_scalar("Eval/total_reward", mean_reward, steps)
        writer.add_scalar("Eval/n_violations", n_violations, steps)
        writer.add_scalar("Eval/n_reached_goal", n_reached_goal, steps)

        writer.flush()
        writer.close()

if __name__ == '__main__':
    main()
   