# ROTDL: Robust Online Tucker Dictionary Learning from Multidimensional Data Streams

We propose a novel online algorithm called ROTDL for the problem of robust tensor tracking under the Tucker format.
ROTDL is not only capable of tracking the underlying Tucker dictionary of multidimensional data streams over time, but also
robust to sparse outliers. The proposed algorithm is specifically designed by using the alternating direction method of multipliers,
block-coordinate descent, and recursive least-squares filtering techniques. 

![Capture](https://user-images.githubusercontent.com/26319211/189215479-bfdf8c3b-3dad-455d-a818-e915f3831189.PNG)



## DEMO

+ Run "demo_xyz.m" for synthetic experiments.


## State-of-the-art algorithms for comparison

+ **STA** and **DTA**: J. Sun, D. Tao, S. Papadimitriou, P. S. Yu, and C. Faloutsos, “[*Incremental
tensor analysis: Theory and applications*](https://dl.acm.org/doi/10.1145/1409620.1409621),” ACM Trans. Knowl. Discov. Data, vol. 2, no. 3, pp. 1–37, 2008.
+ **ATD**: L.T. Thanh et al. "[*Tracking Online Low-Rank Approximations of Higher-Order Incomplete Streaming Tensors*](https://www.techrxiv.org/articles/preprint/Tracking_Online_Low-Rank_Approximations_of_Higher-Order_Incomplete_Streaming_Tensors/19704034)". TechRxiv, 2022.


## Some Results

+ Effect of the noise level

![noise](https://user-images.githubusercontent.com/26319211/189215151-6aa1d295-ff49-44c2-ad1b-3d9acf6f0b3a.PNG)

+ Effect of the time-varying factor

![time-varying](https://user-images.githubusercontent.com/26319211/189215180-9ac4f82a-c375-4afb-a92e-fe424f14a1f3.PNG)


+ Effect of outliers

![outlier](https://user-images.githubusercontent.com/26319211/189215193-5e04a659-6090-47f8-9b1c-f99ae5c46002.PNG)

+ Comparsion

![compare](https://user-images.githubusercontent.com/26319211/189214710-ad640fd5-0510-4c97-9e0f-452a29f17843.PNG)


## Reference

This code is free and open source for research purposes. If you use this code, please acknowledge the following paper.

[1] **L.T. Thanh**, T.T. Duy, K. Abed-Meraim, N. L. Trung and A. Hafiane. “[*Robust Online Tucker Dictionary Learning from Multidimensional Data Streams*](https://)”. **In Proc. 14th APSIPA-ASC**, 2022. [[PDF]](https://drive.google.com/file/d/1A4r3AEOqje6YDKNDX_pZBzJtN_j7Hqmw/view?usp=sharing).



