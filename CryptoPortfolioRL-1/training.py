# env

import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import spaces
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv

class CryptoPortfolioEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, 
                df,
                crypto_dim,
                initial_amount,
                transaction_cost_pct,
                reward_scaling,
                state_space,
                action_space,
                tech_indicator_list,
                folder,
                hedging,
                day = 0):
        self.day = day
        self.df = df
        self.crypto_dim = crypto_dim
        self.initial_amount = initial_amount
        self.transaction_cost_pct =transaction_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list
        self.folder = folder
        self.actions_memory=[[1/self.crypto_dim]*self.crypto_dim]

        # action_space normalization and shape is self.stock_dim
        self.action_space = spaces.Box(low = 0, high = 1,shape = (self.action_space,)) 

        # load data from a pandas dataframe
        self.data = self.df.loc[self.day,:]

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape = (len(self.tech_indicator_list),self.state_space))
        self.state = np.array([self.data[tech].values.tolist() for tech in self.tech_indicator_list])

        self.terminal = False
        self.reward = 0     
                
        # initalize state: inital portfolio return + individual crypto return + individual weights
        self.portfolio_value = self.initial_amount

        # memorize portfolio value each step
        self.asset_memory = [self.initial_amount]

        # memorize portfolio return each step
        self.portfolio_return_memory = [0]
        self.date_memory=[self.data.date.unique()[0]]

        # whether to hedging
        self.hedging = hedging
        
        if self.hedging[0]:
            self.index_busd = self.hedging[1]
        
    def step(self, actions):
        # print(self.day)
        self.terminal = self.day >= len(self.df.index.unique())-1
        # print(actions)

        if self.terminal:
            df = pd.DataFrame(self.portfolio_return_memory)
            df.columns = ['daily_return']
            plt.plot(df.daily_return.cumsum(),'r')
            plt.savefig(self.folder + 'cumulative_reward.png')
            plt.close()
            
            plt.plot(self.portfolio_return_memory,'r')
            plt.savefig(self.folder + 'rewards.png')
            plt.close()

            print("=================================")
            print("begin_total_asset:{}".format(self.asset_memory[0]))           
            print("end_total_asset:{}".format(self.portfolio_value))

            df_daily_return = pd.DataFrame(self.portfolio_return_memory)
            df_daily_return.columns = ['daily_return']
            if df_daily_return['daily_return'].std() !=0:
                sharpe = df_daily_return['daily_return'].mean()/ \
                       df_daily_return['daily_return'].std()
                print("Sharpe: ",sharpe)
            print("=================================")
            
            return self.state, self.reward, self.terminal,{}

        else:
            
            # normalizing actions
            action_df = pd.DataFrame()
            action_df['raw_action'] = actions
            action_df['scaled_action'] = (action_df['raw_action'].rank(method="first")-0.5)/len(action_df)
            actions = action_df['scaled_action'].values

            weights = self.softmax_normalization(actions)

            # mom <= 0 hedgeing
            if self.hedging[0]:
                if self.df.loc[self.day,:]['hedging'].values[0] <= 0:
                    weights = np.array([0]*self.crypto_dim)
                    weights[self.index_busd] = 1

            self.actions_memory.append(weights)

            last_day_memory = self.data

            # load next state
            self.day += 1
            self.data = self.df.loc[self.day,:]
            
            self.state = np.array([self.data[tech].values.tolist() for tech in self.tech_indicator_list])

            # calcualte portfolio return
            # individual stocks' return * weight
            portfolio_return = sum(((self.data.close.values / last_day_memory.close.values)-1)*weights)
            # update portfolio value
            new_portfolio_value = self.portfolio_value*(1+portfolio_return)
            self.portfolio_value = new_portfolio_value

            # save into memory
            self.portfolio_return_memory.append(portfolio_return)
            self.date_memory.append(self.data.date.unique()[0])            
            self.asset_memory.append(new_portfolio_value)

            # the reward is the new portfolio value or end portfolo value

            dts = self.date_memory
            avs = self.asset_memory
            df_value = pd.DataFrame({'date':dts,'account_value':avs})
            df_value['date'] = pd.to_datetime(df_value['date'], format='%Y-%m-%d')
            df_value = df_value.set_index('date')

            self.reward = new_portfolio_value

        return self.state, self.reward, self.terminal, {}

    def reset(self):
        self.asset_memory = [self.initial_amount]
        self.day = 0
        self.data = self.df.loc[self.day,:]
        self.actions_memory=[[1/self.crypto_dim]*self.crypto_dim]
        
        self.state = np.array([self.data[tech].values.tolist() for tech in self.tech_indicator_list])
        
        self.portfolio_value = self.initial_amount
        #self.cost = 0
        #self.trades = 0
        self.terminal = False 
        self.portfolio_return_memory = [0]
        self.date_memory=[self.data.date.unique()[0]]
        print(f"reset at {self.data['date'].values[0]}")

        return self.state
    
    def render(self, mode='human'):
        return self.state
        
    def softmax_normalization(self, actions):
        numerator = np.exp(actions)
        denominator = np.sum(np.exp(actions))
        softmax_output = numerator/denominator
        return softmax_output
    
    def save_asset_memory(self):
        date_list = self.date_memory
        account_value = self.asset_memory
        df_account_value = pd.DataFrame({'date':date_list,'account_value':account_value})
        return df_account_value

    def save_action_memory(self):
        # date and close price length must match actions length
        date_list = self.date_memory
        df_date = pd.DataFrame(date_list)
        df_date.columns = ['date']
        
        action_list = self.actions_memory
        df_actions = pd.DataFrame(action_list)
        df_actions.columns = self.data.ticker.values
        df_actions.index = df_date.date
        #df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        return df_actions

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs
    
def building_environment(env_kwargs, train, trade):
    
    crypto_dimension = len(train.ticker.unique())
    state_space = crypto_dimension
    print(f"Crypto Dimension: {crypto_dimension}, State Space: {state_space}")

    env_kwargs["state_space"] = state_space
    env_kwargs["crypto_dim"] = crypto_dimension
    env_kwargs["action_space"] = crypto_dimension

    e_train_gym = CryptoPortfolioEnv(df = train, **env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()
    
    e_trade_gym = CryptoPortfolioEnv(df = trade, **env_kwargs)
    env_trade, obs_trade = e_trade_gym.get_sb_env()

    return e_train_gym, e_trade_gym, env_train, crypto_dimension, state_space

# agent

import gym

from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

from stable_baselines3 import A2C
from stable_baselines3 import PPO
from stable_baselines3 import TD3
from stable_baselines3.td3.policies import MlpPolicy
from stable_baselines3 import SAC

MODELS = {"a2c": A2C, "ddpg": DDPG, "td3": TD3, "sac": SAC, "ppo": PPO}       

NOISE = {"normal": NormalActionNoise, "ornstein_uhlenbeck": OrnsteinUhlenbeckActionNoise}

class DRLAgent:

    @staticmethod
    def DRL_prediction(model, environment):

        test_env, test_obs = environment.get_sb_env()
        """make a prediction"""
        account_memory = []
        actions_memory = []
        test_env.reset()
        for i in range(len(environment.df.index.unique())):
            action, _states = model.predict(test_obs, deterministic=True)
            account_memory = test_env.env_method(method_name="save_asset_memory")
            actions_memory = test_env.env_method(method_name="save_action_memory")
            test_obs, rewards, dones, info = test_env.step(action)
            if dones[0]:
                print("hit end!")

                break
        return account_memory[0], actions_memory[0]

    def __init__(self, env):
        self.env = env

    def get_model(
        self,
        model_name,
        policy="MlpPolicy",
        policy_kwargs=None,
        model_kwargs=None,
        verbose=1,
    ):
        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")

        if model_kwargs is None:
            model_kwargs = MODEL_KWARGS[model_name]

        if "action_noise" in model_kwargs:
            n_actions = self.env.action_space.shape[-1]
            model_kwargs["action_noise"] = NOISE[model_kwargs["action_noise"]](
                mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
            )
        print(model_kwargs)
        model = MODELS[model_name](
            policy=policy,
            env=self.env,
            tensorboard_log=f"{'.'}/{model_name}",
            verbose=verbose,
            policy_kwargs=policy_kwargs,
            **model_kwargs,
        )
        return model

    def train_model(self, model, tb_log_name, total_timesteps=5000):
        model = model.learn(total_timesteps=total_timesteps, tb_log_name=tb_log_name)
        return model
    
def train_test_agent(train, trade, e_train_gym, e_trade_gym, env_train, agent_name, total_timesteps, folder, MODEL_KWARGS):

    agent = DRLAgent(env = env_train)
    model = agent.get_model(model_name = agent_name, model_kwargs = MODEL_KWARGS[agent_name])
    trained_agent = agent.train_model(model = model, tb_log_name = agent_name, total_timesteps = total_timesteps)

    df_daily_return, df_actions = DRLAgent.DRL_prediction(model=trained_agent, environment = e_trade_gym)

    # saving files
    df_daily_return.to_csv(f'{folder}df_daily_return_{agent_name}_crypto.csv',index=False)
    df_actions.to_csv(f'{folder}df_actions_{agent_name}_crypto.csv')
    trained_agent.save(f'{folder}trained_{agent_name}_crypto.model')

    return df_daily_return, df_actions, trained_agent