import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

#Hyperparameters
learning_rate = 0.0002
gamma         = 0.98

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.data = []
        
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 2)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=0)
        return x
      
    def put_data(self, item):
        self.data.append(item)
        
    def train_net(self):
        R = 0
        self.optimizer.zero_grad()
        for r, prob in self.data[::-1]:
            R = r + gamma * R # 뒤로 갈수록 감마에 제곱
            loss = -torch.log(prob) * R #gradien 어센트여서 -  
            loss.backward()
        self.optimizer.step()
        self.data = []

def main():
    env = gym.make('CartPole-v1')
    pi = Policy()
    score = 0.0
    print_interval = 20
    
    
    for n_epi in range(10000):
        s, _ = env.reset()
        done = False
        
        while not done: # CartPole-v1 forced to terminates at 500 step.
            prob = pi(torch.from_numpy(s).float()) #state가 4차원 벡터 [막대 각도,차 속도,막대 각속도,차 가속도]를 policy pi에 넣어서 확률분포
            m = Categorical(prob) #카테고리칼은 파이토치에서 지원하는 확률분포
            a = m.sample() # action이 나옴 확률에 비례해서 하나를 뽑아줌
            s_prime, r, done, truncated, info = env.step(a.item()) #env에 action을 던져줌 action이 tensor이기에 스칼라형태로 바꾸기 위해 .item
            pi.put_data((r,prob[a])) #reinforce는 에피소드가 끝나여 학습이 가능 그래서 값을 폴리시에 모아놓음 reward와 그때의 action log파이(s,a)
            s = s_prime
            score += r
            
        pi.train_net()
        
        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {}".format(n_epi, score/print_interval))
            score = 0.0
    env.close()
    
if __name__ == '__main__':
    main()
