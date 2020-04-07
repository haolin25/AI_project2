# README

In our project, we used the Elastic Weight Consolidation (EWC) model which approaches the problem from the regularization point of view. During training, to train a second task means that the neural network forgets about the first task, i.e. catastrophic forgetting. The direct solution to this problem, of course, is to 'lock' the weights that are used to solve the first task when training for the new task. Locking the exact parameters however, is somewhat an impossible task. 

<img src="https://www.pnas.org/content/pnas/114/13/3521/F1.large.jpg?width=100&amp;height=00&amp;carousel=1" style="zoom:33%;" title="abc" />



Instead, Elastic Weight Consolidation tries to converge to a point such that both the first task and the second task have a low error. We try to minimize the loss function
$$
L(\theta)= L_B(\theta) + \sum_i \frac{\lambda}{2}F_i (\theta_i - \theta_{A,i} ^*)^2
$$
where $\lambda$ is the importance of the first task (task A) relative to the second task (task B).

Therefore, the EWC method allows us to get reasonable training results under limited resources, which fits perfectly to our scenario. When several tasks share similar weights, EWC lets the network preserve those sets of weights. Only when several tasks are not smiliar to each other does EWC allocate new node weights in the neural network.

For testing purposes, we used the Rotated MNIST Option in https://github.com/facebookresearch/GradientEpisodicMemory. 