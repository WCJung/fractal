In the study proposed by J. Sohl-Dickstein [1], the presented code (https://github.com/Sohl-Dickstein/fractal) can present that the boundary of convergence or divergence for optimization  have a fractal structure according to each value was disclosed due to the fact that DNN training can be sensitive to small changes in hyperparameters. Based on the study of fractal dimension, we intend to give further insights from the theory of dynamical systems [2].

In order to conduct a comparative study on the loss landscape of DNN and ResNets, we implemented the code to confirm the fact that the performance of ResNets is better than that of general DNNs, and the related previous studies are described in detail as follows.

First, we modified the Python code for the shallow neural networks model (with 300 nodes) [3] that we want to train by applying the open source code of the existing linear model to MNIST, and applied the JAX framework for faster operation. As a result, we were able to identify that there are sections where the optimization results fluctuate significantly even with very small changes in learning late and initial weights. 

As we zoom in on the part with severe changes, we can see that it has a more asymmetrical fractal structure, and we can obtain the final fractal dimension based on the fractal dimension obtained by zooming in. In conclusion, when the values ​​of hyperparameters (batch size, epochs, learning rate, etc.) are changed, we confirmed that both train and test losses have a fractal structure.

The following figure shows the results of convergence and divergence for training (left) and test (right) losses according to the settings for the learning method (MNIST, shallow neural networks with 300 nodes and ReLU, SGD, 300 epochs, 100 batch size, learning rate)

![zzz](https://github.com/user-attachments/assets/8b7bddbe-9c90-434f-9dec-3cecd07f3031)![bb123](https://github.com/user-attachments/assets/b96c78b7-f69a-42b1-8761-507231d0b1f5)


The darker the red, the more divergence, the darker the blue, the more convergence. As we configure the high-quality image, the training amount increases rapidly, and the fractal dimension (=box dimension) for the low-quality (256*256 pixels) standard train error was confirmed to be 1.37.



In this way, we have significantly modified the code for applying parallel computing to ResNets learning, and our research outline for parallel computing is as shown in below. 

![cccccccccc](https://github.com/user-attachments/assets/6a718468-4c5c-49f8-9b5c-abd1c0e0eb51)


However, there is an issue that it takes a lot of time to train ResNets for each hyperparameter in high resolutions and calculate detailed fractal dimensions. We currently have time issues for calculation, so we plan to apply the theory of dynamical systems to obtain the fractal dimension in high-dimensional settings.

[1] J. Sohl-Dickstein, The boundary of neural network trainability is fractal, arXiv:2402.06184 (2024).

[2] A. Fathi, Expansiveness, hyperbolicity and Hausdorff dimension, Communications in mathematical physics 126 (1989), 249-262.

[3] Y. Lecun, L. Bottou, Y. Bengio, and P. Haffner, Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278–2324, 1998.
