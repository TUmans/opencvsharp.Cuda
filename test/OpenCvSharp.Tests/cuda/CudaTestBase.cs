using System;
using System.Collections.Generic;
using System.Text;
using OpenCvSharp.Tests;
using Xunit;

namespace OpenCvSharp.Tests.Cuda;

public abstract class CudaTestBase : TestBase
{

    protected void VerifyCudaSupport()
    {
        if (Cv2.Cuda.GetCudaEnabledDeviceCount() == 0) 
            throw new SkipException("No CUDA device available.");
    }
}
