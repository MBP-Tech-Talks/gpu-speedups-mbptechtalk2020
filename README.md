# gpu-speedups-mbptechtalk2020

-Numba tutorial notebooks for MBP tech talk 2020.
-Forked from [pydata-amsterdam2019-numba](https://github.com/ContinuumIO/pydata-amsterdam2019-numba). 
-[Video presentation walking through pydata-amsterdam2019-numba notebooks at PyData 2019](https://pyvideo.org/pydata-amsterdam-2019/create-cuda-kernels-from-python-using-numba-and-cupy.html)

# Dependencies

```
numpy numba cupy jupyter scipy matplotlib
```

# Why GPUs? 
Many computational problems are parallelizable and can be sped up 10-1000 times on the GPU. Suitable computations are doing the same thing to pieces of 1000s - trillions of pieces of data (records, samples, entries, etc). 

Let's think of a familiar example: the average. An average of numbers stored in a vector can be computed by accumulating these numbers in one spot, and then dividing by the total number. The average does not depend on what order they are added together in, and no number "speaks" to another number. A for loop could do this, but why use a loop? Every pass through the loop is independent of each other. What if all the passes through the loops could be done at the same time? If you have problems like these, GPU speedups can pay off.

Python already has libraries for vectorized problems, like numpy. However, much of this code is actually a low level (eg in C) implementation that is non paralleled, but just loop really fast. There is overhead transferring data to the GPU, but if the computation is complicated enough, then doing it on the GPU can be worth it.

Also, some problems are not (easily) vectorizable or written as fast Numpy ufuncs. For example, each sub computation is independent, but you need some if/else clauses and can't put this into a matrix. If you cannot write down the problem as a series of matrix operations, but can write out what to do in each case, then you can code a kernel function that the GPU cores execute. There are 1000s of GPU cores on a single GPU device, unlike dozens of CPUs.
