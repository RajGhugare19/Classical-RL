# [Multi-Armed-Bandit problem](https://en.wikipedia.org/wiki/Multi-armed_bandit)

### Introduction : 
* Multi-armed bandits are a rich area of research on its own but it is often treated by many as an immediate RL problem. It scales down the more intricate details of full RL problems while maintaining the same gist.
* The basic setup of a K-armed bandit problem is as follows :
 
     A = {0,1,2.....K} , Action set or Arm set
* Each time an action or an arm is selected a reward will be generated, sampled from any distribution, but specific to every arm.

    $q_*(a)$ is known as the true value function which returns the expected value of the reward distribution of arms.
    $a^*$ is the true best action.
    
    
### Solutions : 

There are three ways in which we can the think of the solutions for a bandit problem :

1. ##### Asymptotic Correctness : 
      As the name suggests this concept of a solution will guarantee that eventually the agent will reach the optimal arm. Old and naive techniques like Epsilon-greedy and softmax exploration techniques will provide you with an asymptotic solution.
      
2. ##### Regret Optimality : 
      Rather than just asymptotically converging towards the solution this puts up another level of optimality by trying to maximise the likelyhood of reward that we obtain initially while exploring.Many algorithms like UCB and its variants try to maximize the expected reward while exploring.
     
3. ##### Pac optimality : 
      Given an ($\epsilon$,$\delta$) pair, the agent should return an arm with a gaurantee that the value of that arm should be within $\epsilon$ limits of the value of the true best arm with a probablity of 1-$\delta$.In mathematical terms 
 $$pr[q_*(a) \geq q_*(a^*) - \epsilon] \geq 1 - \delta $$
 where $\epsilon$,$\delta$ $\in [0,1]$
 
 ### Implementation :  
 
 Every algorithm was tested with 10 arms having a bernoulli distribution with : 
 $q_*(a) = [0.1, 0.5, 0.7, 0.73, 0.756, 0.789, 0.81, 0.83, 0.855, 0.865]$
 
 `NOTE`
>  You can change the distibution to a gaussian by using the GaussianStationaryBandit imported from bandits class
 
 ### Results : 
 
 ![](https://i.imgur.com/eLYVDfb.png)

This graph does not tell you the complete story as there are mny other aspects of a multi-armed bandit problem like regret,noisy rewards,$(\epsilon,\delta)$ gaurentees etc.
You can easliy run these algorithms by importing my classes and testing them out in various conditions. 