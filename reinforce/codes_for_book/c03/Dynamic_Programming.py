#encoding:utf-8

#异步动态规划进行策略评估、策略迭代和价值迭代，可明显降低计算量，提高计算效率

#1 小型方格世界MDP建模

# 4*4 方格状态命名
# 状态0和15为终止状态
#  0  1  2  3
#  4  5  6  7
#  8  9 10  11
# 12 13 14  15

from utils import policy_evaluate
from utils import display_V
from utils import P
from utils import R
from utils import uniform_random_pi
from utils import greedy_pi
from utils import policy_iterate
from utils import value_iterate
from utils import greedy_policy
from utils import display_policy

S = [i for i in range(16)]
A = ["n", "e", "s", "w"] 

gamma = 1.00
MDP = S, A, P, R, gamma

#2.策略评估

#异步动态规划，使用最近更新的状态价值，更新当前状态价值
V = [0  for _ in range(16)] # 状态价值初始化
V_pi = policy_evaluate(MDP, V, uniform_random_pi, 100)
print("随机策略评估： ")
print("")
display_V(V_pi)

V = [0  for _ in range(16)] # 状态价值
V_pi = policy_evaluate(MDP, V, greedy_pi, 100)
display_V(V_pi)

#3.策略迭代
V = [0  for _ in range(16)] # 重置状态价值
V_pi = policy_iterate(MDP, V, greedy_pi, 1, 100)
print("optimal value: ")
print("")
display_V(V_pi)

#4 价值迭代
print("价值迭代")
V = [0  for _ in range(16)] # 重置状态价值
display_V(V)

V_star = value_iterate(MDP, V, 4)
print("value iteration process: ")
print("")
display_V(V_star)


print("显示状态对应的行为：")
print("")
display_policy(greedy_policy, MDP, V_star)


















