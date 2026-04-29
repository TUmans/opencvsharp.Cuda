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

    [Fact]
    public void EnsureSizeIsEnough_Test()
    {
        VerifyCudaSupport();

        // 1. Start with an empty GpuMat
        using var gpuMat = new GpuMat();
        Assert.True(gpuMat.Empty());

        // 2. Ensure it is 100x100
        Cv2.Cuda.EnsureSizeIsEnough(100, 100, MatType.CV_8UC1, gpuMat);

        Assert.False(gpuMat.Empty());
        Assert.Equal(100, gpuMat.Rows);
        Assert.Equal(100, gpuMat.Cols);
        Assert.Equal(MatType.CV_8UC1, gpuMat.Type());

        // 3. Ensure it is 200x200
        // This will trigger a reallocation
        Cv2.Cuda.EnsureSizeIsEnough(200, 200, MatType.CV_8UC1, gpuMat);

        Assert.Equal(200, gpuMat.Rows);
        Assert.Equal(200, gpuMat.Cols);

        // 4. Call it again with the same 200x200 size
        // This should do nothing (no reallocation) and remain 200x200
        Cv2.Cuda.EnsureSizeIsEnough(200, 200, MatType.CV_8UC1, gpuMat);

        Assert.Equal(200, gpuMat.Rows);
    }

}
