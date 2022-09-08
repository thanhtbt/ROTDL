# ROTDL: Robust Online Tucker Dictionary Learning from Multidimensional Data Streams

We propose a novel online algorithm called ROTDL for the problem of robust tensor tracking under the Tucker format.
ROTDL is not only capable of tracking the underlying Tucker dictionary of multidimensional data streams over time, but also
robust to sparse outliers. The proposed algorithm is specifically designed by using the alternating direction method of multipliers,
block-coordinate descent, and recursive least-squares filtering techniques. 

![Capture](https://user-images.githubusercontent.com/26319211/189214323-f161c908-a8a4-472c-85b8-5b1415e9f2c7.PNG)



## DEMO

+ Run "demo_xyz.m" for synthetic experiments.


## State-of-the-art algorithms for comparison

+ STA and DTA: J. Sun, D. Tao, S. Papadimitriou, P. S. Yu, and C. Faloutsos, “[*Incremental
tensor analysis: Theory and applications*](https://dl.acm.org/doi/10.1145/1409620.1409621),” ACM Trans. Knowl. Discov. Data, vol. 2, no. 3, pp. 1–37, 2008.
+ ATD: L.T. Thanh et al. "[*Tracking Online Low-Rank Approximations of Higher-Order Incomplete Streaming Tensors*](https://www.techrxiv.org/articles/preprint/Tracking_Online_Low-Rank_Approximations_of_Higher-Order_Incomplete_Streaming_Tensors/19704034)". TechRxiv, 2022.


## Some Results

+ Effect of the noise level

![noise](https://user-images.githubusercontent.com/26319211/189214629-29ee0639-2e73-4ec3-90cb-282626ae0357.PNG)

+ Effect of the time-varying factor

![time-varying](https://user-images.githubusercontent.com/26319211/189214668-9149514c-65aa-4f7f-abc2-7f425890e470.PNG)

+ Effect of outliers
![outlier](https://user-images.githubusercontent.com/26319211/189214695-4362ee69-4d6f-47e0-88b8-da2b18b67c8f.PNG)

+ Comparsion

![compare](https://user-images.githubusercontent.com/26319211/189214710-ad640fd5-0510-4c97-9e0f-452a29f17843.PNG)


## References

This code is free and open source for research purposes. If you use this code, please acknowledge the following paper.

[1] **L.T. Thanh**, T.T. Duy, K. Abed-Meraim, N. L. Trung and A. Hafiane. “*Robust Online Tucker Dictionary Learning from Multidimensional Data Streams*”. **In Proc. 14th APSIPA-ASC**, 2022. 



