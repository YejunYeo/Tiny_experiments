

Key Terms:

Thread = single strand of an execution of a kernel
Threadgroup = group of threads that run on the same physical piece of the GPU.
Grid = Group of Threadgroups

So it goes from groups -> Threadgroups -> threads

Why do we need threadgroups?

1. Shared Fast Memory
- Like a CPU cache, each thregroup has its own threadgroup memory which stores frequently accessed data by the threads. In GPU programming, we have to manually write what kind of things to keep in the thregroup memory. 

For example:

```
threadgroup float tile[16];
tile[lid.x] = data[lid.x];                              
threadgroup_barrier(mem_flags::mem_threadgroup);
float a = tile[5];   // fast
float b = tile[5];   // fast


```
```
1st line
In 
threadgroup float tile[16],

the "threadgroup" keyword tells the metal compiler to allocate the float array to the threadgroup memory.

2nd line
tile[lid.x] = data[lid.x];  
The data[lid.x] was established in the kernel signature and grabs the corresponding data from memory to the threadgroup.

It can be tile[lid.x] = data[lid.x + something random]; and it would still work. 
```



2. Synchronization
- When a threadgroup reaches barrier call, the threadgroup waits until all the threads in the threadgroup have also reached that barrier. Only then does the program proceed.

What is a barrier call?
```
threadgroup_barrier(mem_flags::mem_threadgroup); // line B: the barrier
```
It will look something like this.


It just forces everything to pause until all the threads reach the barrier.

For example:

Thread 1 could take 100 cycles to execute the instruction

BUT

Thread 500 could take 400 cycles to execute perhaps because it was scheduled later.

Threads are executed at different times.

```
kernel void blur(...) {
    int i = lid.x;
    
    tile[i] = data[i];                              // line A: write my pixel to scratchpad
    // NO BARRIER HERE
    float result = tile[i-1] + tile[i] + tile[i+1]; // line C: read neighbors
}
```

|Time|Thread 0|Thread 1|
|---|---|---|
|cycle 10|finishes writing `tile[0]`|still waiting for `data[1]`|
|cycle 11|reads `tile[-1] + tile[0] + tile[1]` ⚠️|still waiting|
|cycle 300|(already moved on)|finally writes `tile[1]`|
In standard python scripts, every operation is executed sequentially from top to bottom. But in GPU programming, only EACH thread runs sequentially with different speeds. Barriers force the threads in a threadgroup to regroup before each thread carries out certain operations.


```
#include <metal_stdlib>
using namespace metal;
kernel void E_4_4(device float* data0, device float* data1, device float* data2, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  int lidx0 = lid.x; /* 4 */
  int alu0 = (lidx0<<2);
  float4 val0 = *((device float4*)((data1+alu0)));
  float4 val1 = *((device float4*)((data2+alu0)));
  *((device float4*)((data0+alu0))) = float4((val0.x+val1.x),(val0.y+val1.y),(val0.z+val1.z),(val0.w+val1.w));
} 
```
Metal is just Apple's GPU programming language.



