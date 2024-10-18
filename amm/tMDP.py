# import numpy as np
# import matplotlib.pyplot as plt
# import math

# # delta change of amount of Token A
# def delta_x(p1, p2):
#     return 1 / math.sqrt(p2) - 1 / math.sqrt(p1)

# # delta change of amount of Token B
# def delta_y(p1, p2):
#     return math.sqrt(p2) - math.sqrt(p1)

# # amount of Token A with 1 unit of liquidity in [a, b] bucket at price p
# def bucket_assets_x(a, b, p):
#     if p < a:
#         return delta_x(b, a)
#     elif p <= b:
#         return delta_x(b, p)
#     else:
#         return 0.

# # amount of Token B with 1 unit of liquidity in [a, b] bucket at price p
# def bucket_assets_y(a, b, p):
#     if p < a:
#         return 0.
#     elif p <= b:
#         return delta_y(a, p)
#     else:
#         return delta_y(a, b)

# def B(z1, z2, Pm):
#     return Pm * z1 + z2

# # value of 1 unit of liquidity in [a, b] bucket at price p
# def value_of_liquidity(a, b, p):
#     return p * bucket_assets_x(a, b, p) + bucket_assets_y(a, b, p)

# def Unit_liquidity_Fee(a, b, p1, p2, fee_rate):
#     if (p1 < a and p2 < a) or (p1 > b and p2 > b) or p1 == p2:
#         return 0., 0.
#     if p1 < p2:
#         return 0., fee_rate * delta_y(max(p1, a), min(p2, b))
#     else:
#         return fee_rate * delta_x(min(b, p1), max(p2, a)), 0.

# def calculate_fees(price_sequence, buckets, fee_rate, liquidity_allocations):
#     Fee_A = {bucket: 0 for bucket in buckets}
#     Fee_B = {bucket: 0 for bucket in buckets}
#     for i in range(len(price_sequence) - 1):
#         P_i = price_sequence[i]
#         P_next = price_sequence[i + 1]
#         for j, bucket in enumerate(buckets):
#             a, b = bucket
#             delta_Fee_A, delta_Fee_B = Unit_liquidity_Fee(a, b, P_i, P_next, fee_rate)
#             Fee_A[bucket] += delta_Fee_A * liquidity_allocations[j]
#             Fee_B[bucket] += delta_Fee_B * liquidity_allocations[j]
#     return Fee_A, Fee_B

# def allocate_liquidity(W, n, price_sequence):
#     x = np.random.dirichlet(np.ones(n))
#     liquidity_allocations = [W * x_i for x_i in x]
#     startprice, endprice = price_sequence[0], price_sequence[-1]
#     wi, wi_prime = [], []
#     for bucket in buckets:
#         a, b = bucket
#         w = value_of_liquidity(a, b, startprice)
#         w_prime = value_of_liquidity(a, b, endprice)
#         wi.append(w)
#         wi_prime.append(w_prime)
#     liquidity_allocations = [W * x[i] / wi[i] for i in range(n)]
#     value_position_list = [liquidity_allocations[i] * wi_prime[i] for i in range(n)]
#     value_position = np.sum(value_position_list)
#     return liquidity_allocations, value_position

# def create_price_buckets(start, end, n):
#     step_size = (end - start) / n
#     buckets = []
#     current_a = start
#     for i in range(n):
#         current_b = current_a + step_size
#         buckets.append((current_a, current_b))
#         current_a = current_b
#     return buckets

# # GeometricBrownianMotionPriceModel class
# class GeometricBrownianMotionPriceModel:
#     def __init__(self, mu, sigma):
#         self.mu = mu
#         self.sigma = sigma
#     def get_price_sequence_sample(self, p0, t_horizon):
#         price_seq = [p0]
#         for t in range(t_horizon):
#             price_seq.append(price_seq[-1] * np.exp(self.mu + self.sigma * np.random.normal()))
#         return price_seq

# # Parameters
# mu, sigma = 4.8350904967723856e-08, 0.004411197134608392
# p0 = 100
# t_horizon = 100
# sample_paths = 20
# n = 10
# W = 1000
# start = 90
# end = 110
# fee_rate = 0.03
# model = GeometricBrownianMotionPriceModel(mu, sigma)
# all_price_sequences = [model.get_price_sequence_sample(p0, t_horizon) for _ in range(sample_paths)]

# buckets = create_price_buckets(start, end, n)
# price_sequence = all_price_sequences[0]  # Use first sample for simplicity in this example

# # Allocate and calculate fees at defined intervals
# interval = 25
# total_intervals = t_horizon // interval
# for k in range(total_intervals):
#     sub_sequence = price_sequence[k*interval:(k+1)*interval+1]
#     liquidity_allocations, value_position = allocate_liquidity(W, n, sub_sequence)
#       # Update W with the value position
#     total_Fee_A, total_Fee_B = calculate_fees(sub_sequence, buckets, fee_rate, liquidity_allocations)
#     fee_a = sum(total_Fee_A.values())
#     fee_b = sum(total_Fee_B.values())
#     total_fee = B(fee_a, fee_b, sub_sequence[-1])
#     reward =  value_position + total_fee -W
#     W = value_position + total_fee 
      
#     print(f"Interval {k+1}: Total Fees for Token A: {fee_a}, Total Fees for Token B: {fee_b}, Total Value: {total_fee + value_position},reward: {reward}")













import numpy as np
import matplotlib.pyplot as plt
import math
from collections import defaultdict

# 你自己定义的delta_x, delta_y等函数
def delta_x(p1, p2):
    return 1 / math.sqrt(p2) - 1 / math.sqrt(p1)

def delta_y(p1, p2):
    return math.sqrt(p2) - math.sqrt(p1)

def bucket_assets_x(a, b, p):
    if p < a:
        return delta_x(b, a)
    elif p <= b:
        return delta_x(b, p)
    else:
        return 0.

def bucket_assets_y(a, b, p):
    if p < a:
        return 0.
    elif p <= b:
        return delta_y(a, p)
    else:
        return delta_y(a, b)

def B(z1, z2, Pm):
    return Pm * z1 + z2

def value_of_liquidity(a, b, p):
    return p * bucket_assets_x(a, b, p) + bucket_assets_y(a, b, p)

def Unit_liquidity_Fee(a, b, p1, p2, fee_rate):
    if (p1 < a and p2 < a) or (p1 > b and p2 > b) or p1 == p2:
        return 0., 0.
    if p1 < p2:
        return 0., fee_rate * delta_y(max(p1, a), min(p2, b))
    else:
        return fee_rate * delta_x(min(b, p1), max(p2, a)), 0.

# 根据提供的定义调整计算手续费
def calculate_fees(price_sequence, buckets, fee_rate, liquidity_allocations):
    Fee_A = {bucket: 0 for bucket in buckets}
    Fee_B = {bucket: 0 for bucket in buckets}
    for i in range(len(price_sequence) - 1):
        P_i = price_sequence[i]
        P_next = price_sequence[i + 1]
        for j, bucket in enumerate(buckets):
            a, b = bucket
            delta_Fee_A, delta_Fee_B = Unit_liquidity_Fee(a, b, P_i, P_next, fee_rate)
            Fee_A[bucket] += delta_Fee_A * liquidity_allocations[j]
            Fee_B[bucket] += delta_Fee_B * liquidity_allocations[j]
    return Fee_A, Fee_B

# # 分配流动性函数使用你自己提供的逻辑
# def allocate_liquidity(W, x, price_sequence, buckets):
#     #x = np.random.dirichlet(np.ones(n))
#     print(type(x),price_sequence)
#     # n = len(x)
#     liquidity_allocations = [W * x_i for x_i in x]
#     startprice, endprice = price_sequence[0], price_sequence[-1]
#     wi, wi_prime = [], []
#     for bucket in buckets:
#         a, b = bucket
#         w = value_of_liquidity(a, b, startprice)
#         w_prime = value_of_liquidity(a, b, endprice)
#         wi.append(w)
#         wi_prime.append(w_prime)
#     liquidity_allocations = [W * x[i] / wi[i] for i in range(len(x))]
#     value_position_list = [liquidity_allocations[i] * wi_prime[i] for i in range(len(x))]
#     value_position = np.sum(value_position_list)
#     return liquidity_allocations, value_position


def allocate_liquidity(W, action, price_sequence, buckets):
    # Ensure action is a numpy array
    
    action = np.asarray(action)
    liquidity_allocations = W * action
    startprice = price_sequence[0]  # Assuming price_sequence is list of lists with price at index 0
    endprice = price_sequence[-1] # Adjust according to your structure

    wi, wi_prime = [], []
    for bucket in buckets:
        a, b = bucket
        w = value_of_liquidity(a, b, startprice)
        w_prime = value_of_liquidity(a, b, endprice)
        wi.append(w)
        wi_prime.append(w_prime)
    
    # Prevent division by zero and ensure allocation ratios are calculated properly
    liquidity_allocations = [liquidity_allocations[i] / wi[i] if wi[i] != 0 else 0 for i in range(len(buckets))]
    value_position_list = [liquidity_allocations[i] * wi_prime[i] for i in range(len(buckets))]
    value_position = sum(value_position_list)

    #print(f"Allocations: {liquidity_allocations}, Value Position: {value_position}")
    return liquidity_allocations, value_position



# 环境类定义，用于与智能体交互
class TradingEnvironment:
    def __init__(self, buckets, price_sequence, W_initial, fee_rate):
        self.buckets = buckets
        self.price_sequence = price_sequence
        self.initial_W = W_initial
        self.W = W_initial
        self.fee_rate = fee_rate
        self.state_index = 0
        self.state_size = 2  # 当前财富和价格
        self.action_size = len(buckets)

    def reset(self):
        self.W = self.initial_W
        self.state_index = 0
        return self.get_state()

    def get_state(self):
        price = self.price_sequence[self.state_index]
        return np.array([self.W, price])

    def step(self, action):
        current_price = self.price_sequence[self.state_index]
        next_index = min(self.state_index + 1, len(self.price_sequence) - 1)
        next_price = self.price_sequence[next_index]
        
        # 分配流动性并计算值位置和手续费
        sub_sequence = self.price_sequence[self.state_index:next_index + 1]
        
        liquidity_allocations, value_position = allocate_liquidity(self.W, action, sub_sequence, self.buckets)
        total_Fee_A, total_Fee_B = calculate_fees(sub_sequence, self.buckets, self.fee_rate, liquidity_allocations)
        
        # 计算总费用和奖励
        fee_a = sum(total_Fee_A.values())
        fee_b = sum(total_Fee_B.values())
        total_fee = B(fee_a, fee_b, next_price)
        reward = value_position + total_fee - self.W
        self.W = value_position + total_fee
        
        self.state_index = next_index
        done = self.state_index == len(self.price_sequence) - 1
        next_state = self.get_state()
        return next_state, reward, done

# SARSA 智能体定义
class SARSAAgent:
    def __init__(self, state_size, action_size, alpha=0.01, gamma=0.99, epsilon=1.00, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.Q = defaultdict(lambda: np.zeros((action_size,)))
        

    def choose_action(self, state):
        # if np.random.rand() < self.epsilon:
            
        #     action = np.random.dirichlet(np.ones(self.action_size))  # 随机分配比例作为动作
            #print("Random Action Taken:", action)
        #     return action
        # action = self.Q[tuple(state)]
        #print("Greedy Action Taken:", action)
        action = np.random.dirichlet(np.ones(self.action_size))
        return action

    def update(self, state, action, reward, next_state, next_action):
        current = self.Q[tuple(state)][np.argmax(action)]  # 使用最大动作概率对应的索引
        next_value = self.Q[tuple(next_state)][np.argmax(next_action)]
        target = reward + self.gamma * next_value
        self.Q[tuple(state)][np.argmax(action)] += self.alpha * (target - current)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
# GeometricBrownianMotionPriceModel class
class GeometricBrownianMotionPriceModel:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
    def get_price_sequence_sample(self, p0, t_horizon):
        price_seq = [p0]
        for t in range(t_horizon):
            price_seq.append(price_seq[-1] * np.exp(self.mu + self.sigma * np.random.normal()))
        return price_seq
def create_price_buckets(start, end, n):
    step_size = (end - start) / n
    buckets = []
    current_a = start
    for i in range(n):
        current_b = current_a + step_size
        buckets.append((current_a, current_b))
        current_a = current_b
    return buckets
# 创建交易环境并训练
buckets = create_price_buckets(90, 110, 10)
model = GeometricBrownianMotionPriceModel(mu=4.8350904967723856e-08, sigma=0.004411197134608392)
price_sequence = model.get_price_sequence_sample(p0=100, t_horizon=100)
env = TradingEnvironment(buckets, price_sequence, W_initial=1000, fee_rate=0.03)
agent = SARSAAgent(env.state_size, env.action_size)

# 训练智能体
episodes = 50
for episode in range(episodes):
    state = env.reset()
    action = agent.choose_action(state)
    total_reward = 0

    while True:
        next_state, reward, done = env.step(action)
        next_action = agent.choose_action(next_state)
        
        agent.update(state, action, reward, next_state, next_action)
        state, action = next_state, next_action
        total_reward += reward

        if done:
            break

    if episode % 10 == 0:
        print(f"Episode {episode + 1}: Total Reward: {total_reward}")





