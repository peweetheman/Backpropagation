# Multilayer Perceptron
## Patrick Phillips

In this project I create a two layer neural network and implement back propogation from scratch.

In my code I use the development data to loop through and choose which set of weights over all iterations was optimal, 
this way avoiding overfitting the training data.

Typically as the iterations increased the performance continued to increase, although the neural net is definitely 
slower than a single perceptron so I did not try more than 25 iterations, but I found that after about 10 iterations
there was little or no improvement in the dev data, suggesting that the weights had converged to optimal values already.

![](Accuracy_vs_Iterations.png)

I also experimented with different learn rates and had the most success with consistent and quick convergence to an accuracy of about
85% when I used a lr of .1, I found that sometimes the success % would get stuck at the .7567 accuracy with different combinations of hyperparameters, which indicates that a local maxima was likely found. With higher learning rates over .01 this problem didn't occur, as the updates were significant enough to jump over this peak. I found that overall, the neural network was better than the perceptron. As shown in the graph above, I had results that peaked at roughly 85% accuracy which is about 5% better than the perceptron algorithm was able to do.

