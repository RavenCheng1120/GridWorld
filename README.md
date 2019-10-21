# 1.GridWorld_policy evaluation
強化式機器學習練習-GridWorld    
Matplotlib animation for fixed policy    

> 1. change grid to (10,10)    
> 2. 3 blocks that cannot access    
> 3. show fixed policy     

計算Fixed policy之下的Grid World V[s]值，以Policy Evaluation方法計算。     

以下是主要演算法： 
```python
while True:
    biggest_change = 0.0
    for s in states:                     #s會得到(0, 1)、(1, 2)...等11種state
        old_v = v[s]
        new_v = 0
        if s in grid.actions.keys():
            #for a in grid.actions[s]:   #uniform random
            a = grid.actions[s]          #a會得到R或U
            grid.set_state(s)            #i,j會變成s所在的格子
            r = grid.move(a)             #i,j的數字會隨前進方向改變，r會收到回傳的reward值
            new_v = r + gamma * v[grid.current_state()]
        v[s] = round(new_v,5)
        biggest_change = max(biggest_change, np.abs(old_v - new_v))
    if biggest_change < small_enough:
        break
```    
![image](https://github.com/RavenCheng1120/GridWorld/blob/master/RL_HW2_Fixed%20Policy%20Animation/result.png)    


# 2.GridWorld_policy iteration and value iteration
計算Grid World的V[s]與最佳policy，分為policy iteration以及value iteration      
### policy iteration主要演算法：      
1-policy evaluation部分   
```python
while True:
    biggest_change = 0.0
    for s in states:                     #s會得到(0, 1)、(1, 2)...等11種state
        old_v = v[s]
        if s in policy:
            a = policy[s]                #a會得到L,R,D或U
            grid.set_state(s)            #i,j會變成s所在的格子
            r = grid.move(a)             #i,j的數字會隨前進方向改變，r會收到回傳的reward值
            v[s] = r + gamma * v[grid.current_state()]
        biggest_change = max(biggest_change, np.abs(old_v - v[s]))
    if biggest_change < small_enough:
        break 
```       
2-policy improvement部分   
```python
is_policy_converged = True
for s in states:
    if s in policy:
        old_a = policy[s]
        new_a = None
        best_value = float('-inf')
        # loop through all possible actions to find the best current action
        for a in ALL_POSSIBLE_ACTIONS:   #a will loop through L, R, D, and U
            grid.set_state(s)
            r = grid.move(a)
            V = r + gamma * v[grid.current_state()]
            if V > best_value:
                best_value = V
                new_a = a
        policy[s] = new_a
        if new_a != old_a:
            is_policy_converged = False

if is_policy_converged:
    break
```     
![image](https://github.com/RavenCheng1120/GridWorld/blob/master/RL_HW3_Value%20Iteration/PolicyIteration.jpg)  

### value iteration主要演算法：   
1-policy evaluation(優化)部分    
```python
while True:
    biggest_change = 0
    for s in states:
        old_v = v[s]
        if s in policy:
            best_v = float('-inf')
            for a in ALL_POSSIBLE_ACTIONS:
                grid.set_state(s)
                r = grid.move(a)
                V = r + gamma * v[grid.current_state()]
                if V > best_v:
                    best_v = V
            v[s] = best_v
        biggest_change = max(biggest_change, np.abs(old_v-v[s]))
        
    if biggest_change < small_enough:
        break
```    
2-計算policy部分   
```python
for s in policy.keys():
    best_a = None
    best_value = float('-inf')
    # loop through all possible actions to find the best current action
    for a in ALL_POSSIBLE_ACTIONS:
        grid.set_state(s)
        r = grid.move(a)
        V = r + gamma * v[grid.current_state()]
        if V > best_value:
            best_value = V
            best_a = a
    policy[s] = best_a
```     
![image](https://github.com/RavenCheng1120/GridWorld/blob/master/RL_HW3_Value%20Iteration/ValueIteration.jpg)    
    
# 3.Monte Carlo Prediction
+ First-visit Monte Carlo
+ Monte Carlo Exploring Start
+ Monte Carlo with out Exploring Star
    > on-policy first visit MC
    > off-policy first visit MC
