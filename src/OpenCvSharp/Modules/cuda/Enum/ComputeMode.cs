using System;
using System.Collections.Generic;
using System.Text;

namespace OpenCvSharp.Cuda;

public enum ComputeMode
{
    Default = 0,            // multiple threads can use cudaSetDevice() with this device
    Exclusive = 1,          // only one thread in one process will be able to use cudaSetDevice() with this device
    Prohibited = 2,         // no threads can use cudaSetDevice() with this device
    ExclusiveProcess = 3    // many threads in one process will be able to use cudaSetDevice() with this device
}
