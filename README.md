# cvGPUSpeedup

Every memory read, is an opportunity for compute.

With this idea in mind, this library wants to make OpenCV-CUDA code run faster on the GPU. Especially for typical pre and post processing operations for DL networks.

## What does it offer today?

Crop, resize, basic point wise operations (for instance for normalization), color space conversions, color channel split and pack, and a flexible way to pack all those operations in a single fast kernel.

We additionaly created a CircularTensor object that can add a new image (or data matrix), removing the oldest one, and moving the rest of the images one position. It can also do some processing on the new image, while movig the data of the other images, all in a single kernel. And of course, the user can define which processing will be performed on each new image.

Some of those functionalities can be found in OpenCV-CUDA functions. With cvGPUSpeedup, the functions do not execute the code in the GPU, but return an struct that will contain the parameters and the code. When you have all the operations, the only remaining thing to do, is to call cvGS::executeOperations passing a cuda stream, and the operations as parameters in the order you want them to be executed. The result of the nth operation will be the input of the nth + 1 operation as in a Directed Graph.

In addition to the performance gains, this reduces the number of cv::cuda::GpuMat objects required, since you will only need an input GpuMat object for the first operation, and an output GpuMat object for the last operation. This is just the basic functionality, there are more complex options. Unfortunatly, there is no documentation at this time, so the best way to check what is possible is to look at the source code in the tests folder.

## How does it compare to available CUDA optimization technologies and libraries?

One of the best ways to describe the idea behind cvGPUSpeedup, and specially the underliying Fused Kernel Library, would be:

__It is like a compile-time "CUDA Graphs"__

The main difference being that in our case, the graph is compiled by nvcc and generates an extremely optimized single CUDA Kernel. Not only will we have a single CUDA runtime call like with CUDA Graphs, but additionally we will read once from GPU memory and write once into GPU memory. This can make the code execution up to thousands of times faster.

Some NVIDIA libraries do apply some level of kernel fusion, but do not define a way to fuse code components from their libraries with your own code, or other librarie's code.

Our aim is to demonstrate that it is possible to have a CUDA library ecosystem where different functionalities, from different libraries and computing areas, can be combined in a single very fast CUDA Kernel.

In terms of feature completeness, the Fused Kernel Library is less than 1% complete. It currently has mainly one principal contributor for the design and C++/CUDA implementation, one for the CMAKE side of things, and a pair of sporadic contributors.

Our aim with this repository is to create a code demonstration platform, and an space where to keep adding new ideas, features, and share it with the community to make it as big and useful as possible.
  
## Tested hw/sw
*  Cuda SDK 11.8 and 12.1
*  OS Windows 11 22H2 with drivers from 516.94 to 546.17.
*  Ubuntu 22.04 (both native and under WSL2 enviroment)   
*  Compute capabilities 7.5 (Turing), 8.6 (Ampere), 8.9 (ADA Lovelace)
*  All systems with x86_64 cpu architecture

## Using the library
In order to use it, you need to compile your code, along with cvGPUSpeedup library headers, with nvcc (provided by the CUDA toolkit) and at least C++17 support (this is already set by the cmake project).

You can use the cmake install target to copy the headers to any desired path. You can also use the cmake exported target.

If you want to use the cvGS interface, along with OpenCV, both on windows and linux you will need to build opencv with cuda support. If you want to only use the fk (Fused Kernel) namespace, then you won't need OpenCV at all.

### OpenCV example
Let's see an example in OpenCV:
```cpp
// OpenCV version
void testOpenCV()
{
    constexpr int MAX_DETECTIONS = 50;
    std::array<cv::cuda::GpuMat, MAX_DETECTIONS> crops;
    // Fill the crops array with 50 crops of any size, from a source image.

    cv::Scalar subtract_val(1, 4, 6);
    cv::Scalar divide_val(2, 8, 1);

    cv::cudaStream stream;
    cv::Size resDims(64, 128);
    std::array<std::vector<cv::cuda::GpuMat>, MAX_DETECTIONS> cv_output;
    for (int detection = 0; detection < MAX_DETECTIONS; detection++) {
        for (int i = 0; i < CV_MAT_CN(CV_32FC3); i++) {
            cv_output[detection].emplace_back(resDims, CV_32FC1);
        }
    }

    // All this memory required by OpenCV only
    cv::cuda::GpuMat resized(resDims, CV_8UC3);
    cv::cuda::GpuMat float3Image(resDims, CV_32FC3);
    cv::cuda::GpuMat float3Image2(resDims, CV_32FC3);

    double alpha = 0.5;

   // 50 times 5 kernels, 250 kernels in total
    for (int i = 0; i < MAX_DETECTIONS; i++) {
        cv::cuda::resize(crops[i], resized, resDims, 0., 0., cv::INTER_LINEAR, stream);
        resized.convertTo(float3Image, CV_32FC3, alpha, stream);
        cv::cuda::subtract(float3lmage, subtract_val, float3lmage2, cv::noArray(), -1, stream);
        cv::cuda::divide(float3lmage2, divide_val, float3lmage, 1.0, -1, stream);
        cv::cuda::split(float3Image, cv_output[i], stream);
   }
    stream.waitForCompletion();
}
``` 
### cvGPUSpeedup example
Now, same functionality but with cvGPUSpeedup and kernel execution being 38x times faster:

```c++
// cvGPUSpeedup version
void testcvGPUSpeedup()
{
    constexpr int MAX_DETECTIONS = 50;
    std::array<cv::cuda::GpuMat, MAX_DETECTIONS> crops;
    // Fill the crops array with 50 crops of any size, from a source image.

    cv::Scalar subtract_val(1, 4, 6);
    cv::Scalar divide_val(255, 255, 255);
    
    cv::cudaStream stream;
    cv::Size resDims(64, 128);
    cv::cuda::GpuMat output(MAX_DETECTIONS, resDims.width * resDims.height * CV_MAT_CN(CV_32FC3), CV_32FC1);
    double alpha = 0.5;

    // Asume we got the maximum number of detections
    int activeDetections = 50;
    // single kernel, 38x faster than OpenCV in RTX A2000 12GB
    cvGS::executeOperations(stream,
                            cvGS::resize<CV_8UC3, cv::INTER_LINEAR, MAX_DETECTIONS>(crops,resDims,activeDetections)),
                            cvGS::convertTo<CV_8UC3, CV_32FC3>(),
                            cvGS::multiply<CV_32FC3>(cv::Scalar(alpha,alpha,alpha)),
                            cvGS::substract<CV_32FC3>(substract_val),
                            cvGS::divide<CV_32FC3>(divide_val),
                            cvGS::split<CV_32FC3>(output,resDims)
                            );

    stream.waitForCompletion();
}
```

The cvGPUSpeedup version, will do the same, but with a single CUDA kernel, and execute up to 38x times faster, for 50 crops of an image.

# Benchmarks

The library has some unit tests that can be additionally used as benchmarks. When generating benchmark results, they show always positive speedups ranging from 2x to 10000x (in an RTX A2000). The Speedup is going to be greater the more kernels you are fusing, and the smaller those kernels are in terms of both compute operations and grid size. 
## Variable size crop, resize and normalize

![OpenCV timeline](https://github.com/morousg/cvGPUSpeedup/blob/main/images/NSightSystemsTimeline1.png) 
   
In the image above, we show two NSight Systems timelines, where before the execution of the neural network, we have to do some crops, resize, normalization and split of the color channels. 
In the case of OpenCV-CUDA, despite using the GPU you can see that OpenCV is launching many small kernels. This is wasting a lot of compute time in scheduling and memory accesses. You can even see some small Device to Device copies, which the DL programmers thought they needed.
With cvGPUSpeedup since the syntax is pretty similar to OpenCV, and all the parameters passed are OpenCV types, they managed to do the same operations but in 1/167th of the time, and reduced the amount of memory required in the GPU.

## Temporal tensor (15 images)

![cvGPUSpeedup timeline](https://github.com/morousg/cvGPUSpeedup/blob/main/images/NsightSystemsTimeline2.png) 

In this other case, we are updating a temporal Tensor of 15 images, with a new image that needs to be resized and normalized, and other 14 images that where normalized in previous iterations, that need to be split to planar mode and copied in diferent positions of the temporal Tensor. Some CUDA threads will be doing the normalization, and some others will be just copying the old images, all in parallel.

As you can see, the resulting performance makes the pre-processing virtually free, when before it was more than 25% of the total time for the inference.

# Final words and contact
[Grup Mediapro](https://www.mediapro.tv) uses cvGPUSpeedup in the [AutomaticTV](https://www.automatic.tv) multicam live sports production system.  This product depends on custom Deep Neural Networks. Compared to vanilla OpenCV-CUDA implementation,  we obtained speedups of up to 167x in some cases.

If you are interested in investing in cvGPUSpeedup development for your own usage, please contact <oamoros@mediapro.tv>
