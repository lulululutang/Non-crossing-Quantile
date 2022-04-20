# Non-crossing-Quantile

We propose a nonparametric quantile regression method using deep neural networks with a rectified linear unit penalty function to avoid quantile crossing.

1. main.py is the main body of the approach. It shows the result of NC-CQR figures and the performance of the conformal interval.
2. dnn.py is the neural network initialization.
3. lossfunction.py provides the non-crossing check ioss.
4. gen_uni.py is the data generation step.
