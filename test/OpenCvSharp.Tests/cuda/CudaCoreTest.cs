using System;
using System.Collections.Generic;
using System.Text;
using OpenCvSharp.Cuda;
using OpenCvSharp.Tests;
using Xunit;

namespace OpenCvSharp.Tests.Cuda;

public class CudaCoreTest : CudaTestBase
{

    [Fact]
    public void ConvertFp16Test()
    {
        VerifyCudaSupport();

        // Arrange: 32-bit float (CV_32FC1)
        using var cpuSrc = new Mat(3, 3, MatType.CV_32FC1, new Scalar(3.14f));
        using var gpuSrc = new GpuMat(); gpuSrc.Upload(cpuSrc);

        // Act
        using var gpuDst = new GpuMat();
        
        Cv2.Cuda.ConvertFp16(gpuSrc, gpuDst);
        // Assert
        using var cpuDst = new Mat();
        gpuDst.Download(cpuDst);

        Assert.False(cpuDst.Empty());

        Assert.Equal(MatType.CV_16SC1, cpuDst.Type());
        Assert.Equal(3, cpuDst.Rows);
        Assert.Equal(3, cpuDst.Cols);
    }

}
