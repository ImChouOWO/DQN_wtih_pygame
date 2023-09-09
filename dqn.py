from game import GameEnv 
import sys
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import os
import matplotlib.pyplot as plt



class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),  #輸入層
            nn.ReLU(),                  #激活函數
            nn.Linear(128, output_dim)  #輸出層
        )
        
    def forward(self, state):
        return self.fc(state)

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
# Hyper parameters
LEARNING_RATE = 0.01      #學習率
DISCOUNT_FACTOR = 0.8      #折扣因子
EXPLORATION_MAX = 0.9       #隨機動作的最大概率
EXPLORATION_MIN = 0.01      #隨機動作的最小概率
EXPLORATION_DECAY = 0.01   #探索率衰減
BATCH_SIZE = 64             #重播內存中隨機選擇的經驗數量
MEMORY_SIZE = 10000         #存儲多少先前的經驗



# LEARNING_RATE (學習率):
# 這是一個數值，通常介於0和1之間，它決定了我們在每次更新模型參數時應該更改多少。
# 較高的學習率意味著每次更新參數時更大的更改，而較低的學習率則意味著更小的更改。
# 學習率的選擇對模型的學習速度和最終性能都有影響。



# DISCOUNT_FACTOR (折扣因子):
# 在強化學習中，這稱為 gamma (γ)。
# 它是一個介於0和1之間的數值，用於確定我們該如何考慮未來報酬。較高的值意味著我們更重視未來報酬，而較低的值則意味著我們主要關注即時報酬。

# EXPLORATION_MAX, EXPLORATION_MIN, EXPLORATION_DECAY:
# 這些超參數控制了 epsilon-greedy 策略的行為。
# EXPLORATION_MAX 是初始的探索率（即選擇隨機動作的概率）。
# EXPLORATION_MIN 是探索率的最小值，即使在長時間的訓練後，它也不會低於這個值。
# EXPLORATION_DECAY 是探索率衰減的因子。在每個訓練步驟，探索率都會乘以這個值，使其逐步減少，但不低於 EXPLORATION_MIN。

# BATCH_SIZE:
# 這是從重播內存中隨機選擇的經驗數量，用於訓練模型的每次批次。
# 較大的批次大小可能會增加訓練的穩定性，但同時也會增加計算的需求。

# MEMORY_SIZE:
# 這是重播內存的大小，用於存儲先前的經驗。
# 這是一種稱為 Experience Replay 的技術，允許模型使用過去的經驗來學習，這可以提高學習的穩定性。
# MEMORY_SIZE 決定了可以存儲多少先前的經驗。




# Initialize DQN model and replay memory
dqn_model = DQN(input_dim=5, output_dim=2).to(device)
optimizer = optim.Adam(dqn_model.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

replay_memory = deque(maxlen=MEMORY_SIZE)
exploration_rate = EXPLORATION_MAX



# Initialize DQN model and load from saved state if exists
model_path = "DQN\\model\\dqn_model.pth"
if os.path.exists(model_path):
    print("model exisits")
    dqn_model.load_state_dict(torch.load(model_path))
    dqn_model.eval()  # Set the model to evaluation mode
else:
    print("model not exisits")
optimizer = optim.Adam(dqn_model.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

replay_memory = deque(maxlen=MEMORY_SIZE) #當 deque 的大小超過 maxlen 時，最老的元素將從前端被自動刪除，以便於新的元素可以從後端被添加。
exploration_rate = EXPLORATION_MAX




# Initialize environment
env = GameEnv()
step = 3000
rewards_list = []
# Training loop
for episode in range(step):  # Run for 1000 episodes
    state = np.array(env.reset())
    done = False
    total_reward = 0

    while not done:
        # Select action (epsilon-greedy)
        if np.random.rand() < exploration_rate:
            action = random.randrange(2)
        else:
            q_values = dqn_model(torch.FloatTensor(state).to(device)).detach().cpu().numpy()
            action = np.argmax(q_values)

        # Execute action
        next_state, reward, done, _ = env.step(action)
        next_state = np.array(next_state)
        total_reward += reward

        # Store experience in replay memory
        replay_memory.append((state, action, reward, next_state, done))

        # Train DQN model with a mini-batch of samples  從重播內存中隨機選擇一批樣本:
        # 在強化學習中，每個經驗（或稱作轉換）通常由以下五個元素組成：當前狀態、執行的動作、獲得的獎勵、下一個狀態、以及是否達到終止狀態。

        if len(replay_memory) >= BATCH_SIZE:
            mini_batch = random.sample(replay_memory, BATCH_SIZE)
            states, actions, rewards, next_states, dones = zip(*mini_batch)
            states = torch.FloatTensor(states).to(device)
            actions = torch.LongTensor(actions).to(device)
            rewards = torch.FloatTensor(rewards).to(device)
            next_states = torch.FloatTensor(next_states).to(device)
            dones = torch.FloatTensor(dones).to(device)

            # Compute target Q-values
            target_q_values = rewards + (1 - dones) * DISCOUNT_FACTOR * torch.max(dqn_model(next_states), dim=1)[0]
            # Compute current Q-values
            current_q_values = dqn_model(states).gather(1, actions.unsqueeze(1)).squeeze()

            # Update DQN model
            loss = criterion(current_q_values, target_q_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Update state and exploration rate
        state = next_state
        exploration_rate = max(EXPLORATION_MIN, exploration_rate * EXPLORATION_DECAY)
    rewards_list.append(total_reward)
    print(f"Episode {episode+1}, Total Reward: {total_reward}")
plt.plot(rewards_list)
plt.title("Total Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid(True)
plt.savefig('DQN\\pic\\Reward.png')
torch.save(dqn_model.state_dict(), model_path)
