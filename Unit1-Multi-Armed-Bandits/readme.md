# Multi-Armed bandits(Immediate RL problems)
Multi-armed bandits are ignored by a lot of people who begin studying RL,but I think that it is the best place to gain a strong mathematical foothold and get an idea of how things would work in a RL problem.

### Hello world of Reinforcement learning.

I would strongly advice you guys to go through the resources I am going to list down.These will be enough for theoretically studying bandits(atleast enough to get a basic understanding of immediate RL).

1. [Just go through the introduction from wikipedia.](https://en.wikipedia.org/wiki/Multi-armed_bandit)  
2. [Professor Balaraman Ravindran's RL week 1 and week 2](https://nptel.ac.in/courses/106106143/)  
3. [Sutton and Barton chapter 2](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)

As you are watching the lectures it is a good idea that you code the algorithms as you learn them.

I have implemented the following algorithms in agents.py
- [X] epsilon-greedy
- [x] softmax
- [x] UCB1
- [x] Median elimination
- [ ] Other variants of UCB
- [ ] Thompson sampling
- [ ] Policy gradient methods

#The ones which I havent implemented(You could/should if you want to).

After you are done with this I would recommend to go through the notes that I made in the following order.These notes summarize the topics in a brief manner.I would suggest you to make similar notes
* [Overview of the Multi-armed bandit problem](https://hackmd.io/CZQq2azUTMCjt2FF_TQNfQ?view)
* [Regret optimality with UCB1](https://hackmd.io/-DkQQy8DRYezVXDqUaPsYQ)
* [PAC bounds with median elimination](https://hackmd.io/saK7DdqCRnyBfN3HykLhlA)
