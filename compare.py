import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time # Thêm thư viện time để hãm phanh lúc demo

def moving_average(a, n=50):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def epsilon_greedy(Q, state, epsilon, env):
    if np.random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q[state, :])

def train_q_learning(env, episodes=2500, alpha=0.1, gamma=0.99):
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    rewards = []
    
    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        epsilon = max(0.01, 1.0 - ep / (episodes * 0.8))
        
        while True:
            action = epsilon_greedy(Q, state, epsilon, env)
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            best_next_action = np.argmax(Q[next_state, :])
            td_target = reward + gamma * Q[next_state, best_next_action] * (not terminated)
            Q[state, action] += alpha * (td_target - Q[state, action])
            
            state = next_state
            total_reward += reward
            if terminated or truncated:
                break
        rewards.append(total_reward)
    return Q, rewards # <-- TRẢ VỀ THÊM Q-TABLE (Bộ não)

def train_monte_carlo(env, episodes=2500, alpha=0.1, gamma=0.99):
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    rewards = []
    for ep in range(episodes):
        state, _ = env.reset()
        episode_data = []
        total_reward = 0
        epsilon = max(0.01, 1.0 - ep / (episodes * 0.8))
        
        while True:
            action = epsilon_greedy(Q, state, epsilon, env)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_data.append((state, action, reward))
            state = next_state
            total_reward += reward
            if terminated or truncated:
                break
        rewards.append(total_reward)
        
        G = 0
        for t in reversed(range(len(episode_data))):
            s_t, a_t, r_t = episode_data[t]
            G = gamma * G + r_t
            Q[s_t, a_t] += alpha * (G - Q[s_t, a_t])
            
    return Q, rewards

# ==========================================
# PHẦN CHÍNH
# ==========================================
if __name__ == "__main__":
    # BƯỚC 1: TRAIN NGẦM KHÔNG GIAO DIỆN
    env_train = gym.make("Taxi-v4") 
    episodes = 2500
    
    print("Đang huấn luyện Q-Learning (TD)...")
    Q_qlearning, q_rewards = train_q_learning(env_train, episodes)
    
    print("Đang huấn luyện Monte Carlo...")
    Q_mc, mc_rewards = train_monte_carlo(env_train, episodes)
    env_train.close()
    
    # BƯỚC 2: VẼ BIỂU ĐỒ
    print("Đang mở biểu đồ. HÃY TẮT CỬA SỔ BIỂU ĐỒ ĐỂ CHẠY GAME DEMO!")
    plt.figure(figsize=(10, 6))
    plt.plot(moving_average(q_rewards), label="Q-Learning (TD)", color="blue")
    plt.plot(moving_average(mc_rewards), label="Monte Carlo", color="red")
    plt.title("So sánh TD Learning và Monte Carlo (Taxi-v4)")
    plt.xlabel("Số ván chơi (Episodes)")
    plt.ylabel("Phần thưởng trung bình (Total Reward)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show() # CODE SẼ DỪNG Ở ĐÂY CHỜ BẠN TẮT BIỂU ĐỒ
    
    # BƯỚC 3: MỞ GAME DEMO BẰNG BỘ NÃO Q-LEARNING
    print("Đang mở Game Demo...")
    # Bật render_mode="human" để hiện game
    env_demo = gym.make("Taxi-v4", render_mode="human")
    
    for ván in range(3): # Chạy thử 3 ván
        print(f"Bắt đầu ván demo thứ {ván + 1}...")
        state, _ = env_demo.reset()
        
        while True:
            # Lấy bộ não Q_qlearning ra dùng, KHÔNG khám phá ngẫu nhiên nữa (epsilon=0)
            action = np.argmax(Q_qlearning[state, :]) 
            
            state, reward, terminated, truncated, _ = env_demo.step(action)
            
            # Hãm phanh 0.3 giây mỗi bước để mắt người kịp nhìn
            time.sleep(0.3)
            
            if terminated or truncated:
                time.sleep(1) # Dừng 1 chút trước khi sang ván mới
                break
                
    env_demo.close()
    print("Kết thúc Demo!")