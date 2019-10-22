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
## + First-visit Monte Carlo    
*筆記：*
蒙地卡羅預測是在沒有完整環境資訊的情況下使用。    
First-visit MC prediction 是在固定的policy之下，找到各個state的value。    
    1. 先選定隨機起始點，遊玩一次地圖，走到終點算是遊戲結束，此為一個episode。    
    2. 從終點回推各state會得到的G值，把對應state和return(G值)記錄下來。    
    3. 將這個episode中各state第一次出現時的return存起來。    
    4. 重複上述動作許多次，最後將每個state存起來的很多return值平均，就是value值。    
       
## + Monte Carlo Exploring Start    
*筆記：*    
policy是隨機決定，因此會有走向牆壁或無法行走的方塊的情況，為了避免無限迴圈，在玩遊戲時有防禦機制(再度走到同一格時，reward變很低)。    
跟上一個First-visit Monte Carlo程式碼很像，不同處在於加入action影響return，且需要找到最佳policy。    
要通過max_dict(d)找到每個state在該v[s]之下，最佳的前進方向(policy)。循環多次(2000次)後，就能收斂到最佳解法。    
    
    
## + Monte Carlo with out Exploring Star
    > on-policy first visit MC    
    > off-policy first visit MC     

*筆記：*
on-policy不再使用exploring start。
先初始一個隨機的policy，他每次固定從一個起點出發(2,0)，當前進時，有一定機率會往別的方向走。
當下一步是走向迷宮外或是不能走的格子時，他會一直在原地不動，直到觸發隨機往別的方向走，才能離開那個撞牆迴圈。
