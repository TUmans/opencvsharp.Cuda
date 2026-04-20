# OpenCvSharp Cuda support

THIS PROJECT IS STILL WIP.


This fork is an unoffical attempt to bring Cuda support to opencvsharp. 
My goal is to build one working version. Later updates can be verified later on.

What is not my goal : providing ready to go dlls. You'll have to download this & build it yourself. Luckily for you, I made it as easy as possible.
In time, I will submit a merge request with opencvsharp.

## Prerequisite

* **Imporant** : Install latest nvidia driver for your GPU. 
* install visual studio with module "desktop c++ development"
* install cmake
* install git?


We also need the relevant DLL's. You can download them at:

* Windows
	* Nvidia CUDA 12.8 Dlls. [Download](https://developer.nvidia.com/cuda-12-8-0-download-archive)
	* Nvidia CuDnn 9.20 Dlls. [Download](https://developer.nvidia.com/cudnn-9-2-0-download-archive)
	* Nvidia video codex 13.0.37.  [Download](https://developer.nvidia.com/video-codec-sdk#section-get-started)
* Linux
	* WIP


## Generate OpenCvSharp WITH cuda support

* download this repository & pull submodules
* Build OpenCV with cuda support enabled. This requires dlls to be in the right directory. See cmake/opencv_build_options_cuda.cmake for relevant paths.
	* run build_opencv_windows.cuda.ps1
* Build OpenCvSharpExternal.
	* run build_opencvsharpextern.ps1
* Open project & build 

## Deploy your application

To deploy your application, you don't need to install all nvidia software on each device. 
You can bundle your application with dlls from : 
* C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin
* C:\Program Files\NVIDIA\CUDNN\v9.20\bin\12.9\x64
* opencv dlls ( \opencv_artifacts\x64\vc??\lib )
* opencvsharpexternal dll (  \src\build\OpenCvSharpExtern\Release )
* opencvsharp dll ( \src\OpenCvSharp\bin\Release\netstandard2.1 )

Normally you should only make sure your deployment target has a nvidia driver that is up to date enough.

The current build is made to be compatible with RTX30xx series. Update the architecture in cmake/opencv_build_options_cuda.cmake to target other devices.

# Guides

I highly recommend to look at the cuda tests to get a feel of how everything works. 

# Development notes

Using regulary OpenCVSharp functions use CPU processes. If you call a CPU process on a GpuMat it will fail silently. Don't do that. WIP

# OpenCVSharp

[The original OpenCv Readme](README.original.md)
