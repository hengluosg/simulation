import numpy as np
import matplotlib.pyplot as plt
import math

# delta change of amount of Token A
def delta_x(p1, p2):
    return 1 / math.sqrt(p2) - 1 / math.sqrt(p1)


# delta change of amount of Token B
def delta_y(p1, p2):
    return math.sqrt(p2) - math.sqrt(p1)


# amount of Token A with 1 unit of liquidity in [a, b] bucket at price p
def bucket_assets_x(a, b, p):
    if p < a:
        # price below bucket range, assets entirely in Token A
        return delta_x(b, a)
    elif p <= b:
        # price in bucket range, assets in Token A and Token B
        return delta_x(b, p)
    else:
        # price above bucket range, assets entirely in Token B
        return 0.


# amount of Token B with 1 unit of liquidity in [a, b] bucket at price p
def bucket_assets_y(a, b, p):
    if p < a:
        # price below bucket range, assets entirely in Token A
        return 0.
    elif p <= b:
        # price in bucket range, assets in Token A and Token B
        return delta_y(a, p)
    else:
        # price above bucket range, assets entirely in Token B
        return delta_y(a, b)




def B(z1, z2, Pm):
    
    return Pm * z1 + z2


# value of 1 unit of liquidity in [a, b] bucket at price p, converted in units of Token B
def value_of_liquidity(a, b, p): #w_i
    
    return p * bucket_assets_x(a, b, p) + bucket_assets_y(a, b, p)




# transaction fee collected for a single price change by 1 unit of liquidity over [a, b]
def Unit_liquidity_Fee(a, b, p1, p2, fee_rate):
    # return token A, token B amounts
    if (p1 < a and p2 < a) or (p1 > b and p2 > b) or p1 == p2:
        return 0., 0.
    if p1 < p2:
        return 0., fee_rate * delta_y(max(p1, a), min(p2, b))
    else:
        return fee_rate * delta_x(min(b, p1), max(p2, a)), 0.

def calculate_fees(price_sequence, buckets, fee_rate,liquidity_allocations):
    Fee_A = {bucket: 0 for bucket in buckets}  # Initialize Fee_A for each bucket
    Fee_B = {bucket: 0 for bucket in buckets}  # Initialize Fee_B for each bucket
    
    # Loop through each price point in the sequence except the last
    for i in range(len(price_sequence) - 1):
        P_i = price_sequence[i]
        P_next = price_sequence[i + 1]
        
        # Loop through each bucket
        for j, bucket in enumerate(buckets):
            a, b = bucket  # Unpack the tuple defining the bucket
            
            # Calculate fees for Token A and Token B in this bucket
            delta_Fee_A, delta_Fee_B = Unit_liquidity_Fee(a, b, P_i, P_next, fee_rate)
            
            # Update the fees in the dictionary
            Fee_A[bucket] += delta_Fee_A * liquidity_allocations[j]
            Fee_B[bucket] += delta_Fee_B * liquidity_allocations[j]

    return Fee_A, Fee_B

def allocate_liquidity(W, n, price_sequence):
    
    # Generate random proportional allocations that sum to 1
    x = np.random.dirichlet(np.ones(n))
    liquidity_allocations = [W * x_i for x_i in x]
    startprice , endprice = price_sequence[0] , price_sequence[-1]
    # Assuming uniform weights for simplicity, wi can be taken as 1 for each i
    wi ,wi_prime= [] , []
    for bucket in buckets:
        a, b = bucket  # Unpack the tuple defining the bucket

        w = value_of_liquidity(a, b, startprice) #w_i
        w_prime = value_of_liquidity(a, b, endprice)
        wi.append(w) 
        wi_prime.append(w_prime)
    # Compute the absolute liquidity for each bucket

   

    liquidity_allocations = [W * x[i] / wi[i] for i in range(n)]  #l_i
    
    value_position_list = [liquidity_allocations[i]*wi_prime[i]  for i in range(n)] 
    value_position = np.sum(value_position_list)
    return liquidity_allocations , value_position




def create_price_buckets(start, end, n):
    
    step_size = (end - start) / n  # calculate the step size based on the range and number of buckets
    buckets = []

    current_a = start
    for i in range(n):
        current_b = current_a + step_size
        buckets.append((current_a, current_b))
        current_a = current_b  # Ensuring that b_i = a_{i+1}
    
    return buckets


# 定义 GeometricBrownianMotionPriceModel 类
class GeometricBrownianMotionPriceModel:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def get_price_sequence_sample(self, p0, t_horizon):
        price_seq = [p0]
        
        for t in range(t_horizon):
            price_seq.append(price_seq[-1] * np.exp(self.mu + self.sigma * np.random.normal()))
        return price_seq








mu , sigma = 4.8350904967723856e-08, 0.004411197134608392
p0 = 100  
t_horizon = 100   #  T

sample_paths = 20

model = GeometricBrownianMotionPriceModel(mu, sigma)


all_price_sequences = [model.get_price_sequence_sample(p0, t_horizon) for _ in range(sample_paths)]


batch_idx_list = np.random.choice(len(all_price_sequences ), 1, replace=False)

print(batch_idx_list )

plt.figure(figsize=(10, 6))
for price_sequence in all_price_sequences:  
    plt.plot(price_sequence)
plt.title("Sample Paths of Geometric Brownian Motion")
plt.xlabel("Time Steps")
plt.ylabel("Price")
plt.show()

# Example usage:
start = 90  # starting price
end = 110    # ending price
n = 10       # number of buckets
W = 1000
buckets = create_price_buckets(start, end, n)

for i, bucket in enumerate(buckets):
    print(f"Bucket {i}: {bucket}")



fee_rate = 0.03  # Fee rate




price_sequence = all_price_sequences[batch_idx_list[0]]

liquidity_allocations , value_position =  allocate_liquidity(W, n, price_sequence)
print(liquidity_allocations , value_position)


# Calculate fees
total_Fee_A, total_Fee_B = calculate_fees(price_sequence, buckets, fee_rate, liquidity_allocations)




fee_a = sum(total_Fee_A.values())  #the number of fee A
fee_b = sum(total_Fee_B.values())  #the number of fee B


total_fee = B(fee_a , fee_b , price_sequence[-1])
print("Total Fees for Token A:", fee_a)
print("Total Fees for Token B:", fee_b)
total =  total_fee + value_position

print("total", total)
