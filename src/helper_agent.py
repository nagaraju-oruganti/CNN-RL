import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from helper_env import PortfolioEnvironment
from helper_models import BollingerClassifier, PolicyNetwork
from helper_dataset import RLDataset

import warnings
warnings.filterwarnings('ignore')

from collections import deque, namedtuple
import random
import pickle
from tqdm import tqdm

Transition = namedtuple('Transition', ('state', 'norm_state', 
                                       'images', 'labels', 'action', 
                                       'next_state', 'next_norm_state', 'next_images', 'next_labels',
                                       'reward', 'done'))

class DQLAgent:
    def __init__(self, config, name):

        self.config = config
        
        # Unwrap
        self.input_dim = config.input_dim
        self.output_dim = config.output_dim
        self.learning_rate = config.learning_rate
        self.gamma = config.gamma
        self.epsilon = config.epsilon
        self.epsilon_decay = config.epsilon_decay
        self.epsilon_min = config.epsilon_min
        self.reply_capacity = config.reply_capacity
        self.batch_size = config.batch_size
        self.hidden_size = config.hidden_size
        self.riskfree_rate = config.riskfree_rate
        
        ## DEFINE NETWORKS
        self.bollinger_classifier = BollingerClassifier(num_classes = 3)
        self.bollinger_classifier.load_state_dict(
            torch.load(config.base_model_path, map_location=torch.device('cpu'))['model_state_dict']
            )
        self.bollinger_classifier.eval()
        self.policy_network = PolicyNetwork(input_size= self.input_dim, 
                                            hidden_size=self.hidden_size, 
                                            output_size=self.output_dim)
        self.target_network = PolicyNetwork(input_size= self.input_dim, 
                                            hidden_size=self.hidden_size, 
                                            output_size=self.output_dim)
        
        self.bollinger_classifier.to(self.config.device)
        self.policy_network.to(self.config.device)
        self.target_network.to(self.config.device)
        self.policy_network.train()
        
        # very small change because the network is already trained on training dataset
        self.cls_optimizer = optim.Adam(self.policy_network.parameters(), lr = 1e-5)        
        self.rl_optimizer = optim.Adam(self.policy_network.parameters(), lr = self.learning_rate)
        self.loss_function = nn.MSELoss()
        
        self.reply_memory = deque(maxlen = self.reply_capacity)
        self.results = []
        
    def select_action(self, state, norm_state, image, label):
        # Exploration
        if np.random.rand() <= self.epsilon:
            self.explo_count += 1
            weights = np.random.uniform(0, 1, size = self.output_dim)
            trade_signals = np.random.choice([0, 1, 2], size = self.output_dim)       # 0: sell, 1: buy, 2: hold
            weights /= np.sum(weights)
            return weights, trade_signals
        
        # Greedy
        self.model_count += 1
        #state_tensor = torch.tensor(state, dtype = torch.float32)
        logits, inputs = self.construct_inputs_for_one_sample(norm_state, image, label)
        q_values = self.policy_network(inputs)
        weights = q_values.detach().to('cpu').numpy()
        weights /= np.sum(weights)
        
        # convert logits to trade signals
        _, trade_signals = torch.max(logits.to('cpu'), dim=1)
        trade_signals = trade_signals.numpy()
        return weights, trade_signals
    
    # Store
    def store_transition(self, transition):
        self.reply_memory.append(transition)
        
    # Sample transitions
    def sample_transitions(self, batch_size, randomize = True):
        batch = random.sample(self.reply_memory, batch_size)
        batch = Transition(*zip(*batch))
        return batch
    
    
    #### TRAIN
    def train(self, env, episodes):
        for episode in range(episodes):
            state, norm_state, images, labels = env.get_observation()
            env.reset()
            done = False
            total_reward = 0
            episode_returns = []
            self.explo_count = 0
            self.model_count = 0
            
            que = tqdm(range(env.total_steps-1), total = env.total_steps-1, desc = f'episode: {episode}')
            while not done:
                weights, trade_signals = self.select_action(state, norm_state, images, labels)
                next_state, next_norm_state, next_images, next_labels = env.get_observation()
                balance, port_return, reward, done = env.step(weights, trade_signals)
                transition = Transition(state, norm_state, images, labels, weights, next_state, next_norm_state, next_images, next_labels, reward, done)
                self.store_transition(transition)
                
                state, norm_state, images, labels = next_state, next_norm_state, next_images, next_labels
                if len(self.reply_memory) >= self.batch_size:
                    batch = self.sample_transitions(self.batch_size, randomize = np.random.rand() >= 0.5)
                    loss = self.update_q_network(batch)
                    
                total_reward += port_return
                episode_returns.append(port_return)
                
                que.set_postfix({'balance': int(balance),
                                 'total_reward': round(int(balance)/self.config.initial_balance, 4)})
                que.update(1)
                
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
                
            if (episode + 1) % 10 == 0:
                self.target_network.load_state_dict(self.policy_network.state_dict())
                
            que.close()
            annualized_return = 100*((balance / self.config.initial_balance)**(1/4) - 1)        # initial investment is a million
            print(f'Episode: {episode}, epsilion: {self.epsilon:.4f}, loss: {loss:.4f}, reward: {total_reward:.4f}, balance: {int(balance)}, ann_rtn: {annualized_return:.4f}, expo: {self.explo_count}, greedy: {self.model_count}')
            
            self.results.append((episode + 1, self.epsilon, loss, total_reward, balance, annualized_return, self.explo_count, self.model_count))
            
            if (episode + 1) % 50 == 0:
                save_model(self.bollinger_classifier, self.dest_path, name = f'bollinger_cls_{self.name}')
                save_model(self.policy_network, self.dest_path, name = f'dqn_{self.name}')
                save_resutls(self.results, self.dest_path, name = f'results_{self.name}')
                
            ## Save at the end
            save_model(self.bollinger_classifier, self.dest_path, name = f'bollinger_cls_{self.name}')
            save_model(self.policy_network, self.dest_path, name = f'dqn_{self.name}')
            save_resutls(self.results, self.dest_path, name = f'results_{self.name}')
            
        return self.results
    
    def construct_inputs(self, norm_states, images, labels):
        inputs = []
        total_cls_loss = 0
        for norm_state, img, lbl in zip(norm_states, images, labels):
            norm_state = torch.tensor(norm_state, dtype = torch.float32, device = self.config.device)
            m_images = torch.tensor(img, dtype = torch.float32, device = self.config.device)
            m_labels = torch.tensor(lbl, dtype = torch.long, device = self.config.device)
            x, logits, cls_loss = self.bollinger_classifier(m_images, m_labels)
            inputs.append(torch.cat((norm_state.unsqueeze(1), logits), dim=1).view(-1))

            # Accumulate the classification loss for the current sample
            total_cls_loss += cls_loss
        
        total_cls_loss /= len(images)
        inputs = torch.stack(inputs)
        return total_cls_loss, inputs
    
    def construct_inputs_for_one_sample(self, norm_state, image, label):
        norm_state = torch.tensor(norm_state, dtype = torch.float32, device = self.config.device)
        m_images = torch.tensor(image, dtype = torch.float32, device = self.config.device)
        m_labels = torch.tensor(label, dtype = torch.long,device = self.config.device)
        x, logits, cls_loss = self.bollinger_classifier(m_images, m_labels)
        return logits, torch.cat((norm_state.unsqueeze(1), logits), dim=1).view(-1)
        
    ## UDPATE
    def update_q_network(self, batch):
        states = torch.tensor(batch.state, dtype = torch.float32)
        norm_states = torch.tensor(batch.norm_state, dtype = torch.float32)
        weights = torch.tensor(batch.action, dtype = torch.float32)
        next_states = torch.tensor(batch.next_state, dtype = torch.float32)
        next_norm_states = torch.tensor(batch.next_norm_state, dtype = torch.float32)
        rewards = torch.tensor(batch.reward, dtype = torch.float32)
        dones = torch.tensor(batch.done, dtype=torch.float32)
        
        cls_loss, inputs = self.construct_inputs(norm_states, batch.images, batch.labels)
        q_values = self.policy_network(inputs)
        
        _, next_inputs = self.construct_inputs(next_norm_states, batch.next_images, batch.next_labels)
        next_q_values = self.target_network(next_inputs)
        
        targets = rewards + self.gamma * torch.max(next_q_values.to('cpu'), dim = 1)[0]
        targets = targets.unsqueeze(1)
        q_targets = q_values.to('cpu').clone()
        q_targets.scatter_(1, torch.arange(self.output_dim).unsqueeze(0).expand(q_targets.shape[0], -1), weights)
        
        self.rl_optimizer.zero_grad()
        #self.cls_optimizer.zero_grad()
        
        loss = self.loss_function(q_values.to(self.config.device), q_targets.to(self.config.device))
        #cls_loss = 0.9 * cls_loss + 0.1 * loss
        
        #cls_loss.backward(retain_graph=True)
        loss.backward()
        
        self.rl_optimizer.step()
        #self.cls_optimizer.step()
        
        return loss.item()

if __name__ == '__main__':
    from helper_config import Config
    import datetime
    import pandas as pd
    cfg = Config()
    cfg.data_dir = 'data'
    cfg.models_dir = 'models/rl'
    cfg.base_model_path = 'models/cnn_predictor/baseline/model.pth'
    target_assets = len(cfg.tickers)
    
    ### RL / POLICY NETWORK CONFIGURATION
    cfg.input_dim       = len(cfg.tickers) * (4)            # 3 logits from CNN network and 1 norm close price
    cfg.output_dim      = len(cfg.tickers)
    cfg.learning_rate   = 1e-3
    cfg.gamma           = 0.99
    cfg.epsilon         = 0.5
    cfg.epsilon_decay   = 0.995
    cfg.epsilon_min     = 0.01
    cfg.reply_capacity  = 10000
    cfg.batch_size      = 32
    cfg.hidden_size     = 512
    cfg.riskfree_rate   = 0.02
    
    cfg.initial_balance = 1000000    
    
    prep = RLDataset(cfg)
    df_norm_close, df_prices, images, labels = prep.prepare(kind = 'train')
    
    env = PortfolioEnvironment(cfg, df_norm_close.head(50), df_prices.head(50), images, labels, window_size = 2000)
    agent = DQLAgent(cfg, name = 'baseline')
    results = agent.train(env, episodes = 10)