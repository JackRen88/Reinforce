
import numpy as np

i_to_n = {}
i_to_n["0"] = "C1"
i_to_n["1"] = "C2"
i_to_n["2"] = "C3"
i_to_n["3"] = "Pass"
i_to_n["4"] = "Pub"
i_to_n["5"] = "FB"
i_to_n["6"] = "Sleep"

n_to_i = {}

for i, name in zip(i_to_n.keys(), i_to_n.values()):
    n_to_i[name] = int(i)
    
print(n_to_i)

#   C1   C2   C3  Pass Pub   FB  Sleep
Pss = [
   [ 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0 ],
   [ 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.2 ],
   [ 0.0, 0.0, 0.0, 0.6, 0.4, 0.0, 0.0 ],
   [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 ],
   [ 0.2, 0.4, 0.4, 0.0, 0.0, 0.0, 0.0 ],
   [ 0.1, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0 ],
   [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 ]
]    

Pss = np.array(Pss)
print(Pss)

rewards = [-2, -2, -2, 10, 1, -1, 0]

gamma = 0.5

chains =[
    ["C1", "C2", "C3", "Pass", "Sleep"],
    ["C1", "FB", "FB", "C1", "C2", "Sleep"],
    ["C1", "C2", "C3", "Pub", "C2", "C3", "Pass", "Sleep"],
    ["C1", "FB", "FB", "C1", "C2", "C3", "Pub", "C1", "FB",\
     "FB", "FB", "C1", "C2", "C3", "Pub", "C2", "Sleep"]
]

print(len(chains[0]))

def compute_return(start_index = 0,chain = None,gamma = 0.5):
    '''
    '''
    retrn,power,gamma = 0.0 ,0,gamma
    for i in range(start_index,len(chain)):
        retrn += np.power(gamma,power) * rewards[n_to_i[chain[i]]]
        power += 1
    return retrn

result = compute_return(0, chains[3], gamma = 0.5)   
print(result)

result = compute_return(3, chains[3], gamma = 0.5)  
print(result)

def compute_value(Pss,rewards,gamma = 0.05):
    '''
    matrix inverse to compute state value 
    ''' 
    rewards = np.array(rewards).reshape((-1,1))
    # print(rewards)
    values = np.dot(np.linalg.inv(np.eye(Pss.shape[0],Pss.shape[1]) -  gamma * Pss), rewards)
    return values

values = compute_value(Pss, rewards, gamma = 0.999999)
print(values)



    