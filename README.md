# GridWorld_policy evaluation
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
