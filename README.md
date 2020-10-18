# PythonPlaysSnake
I wanted to try a few common reinforcement learning techniques. I thought the snake game was a good example that had not been done to death (although this is definitely not the first snake playing RL repo on github). 

I ended up writing my own simple snake environment using numpy and opencv. I got the best results from Qlearning and Actor Critic using an ANN. Both of these used a state vector that I created to interpret the current state of the game. The vector is essentially the immediate surroundings of the snake's head and the direction of the food. The former took about 10 minutes to train, whereas the later took about 3 hours on my laptop.

Fair warning, training the CNN proved to be very finicky! After almost a day of training on a GPU, I was able to get the CNN snake (which just ingests the game image) to be almost as good as the two methods mentioned above, but since this is just a hobby project, I did not have the time or resources to make it any better. 

The best game I observed was a snake that reached a length 72 from the Actor Critic snake using an ANN. One major difficulty that I think deserves more thought is that once the snake exceeds a certain length, there are adversarial food placements that will always force the snake to eat itself. Perhaps after a sufficient amount of training, a CNN could develop a deep enough representation of the game state to somehow prevent this.
