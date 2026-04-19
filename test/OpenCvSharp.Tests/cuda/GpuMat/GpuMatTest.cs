using System;
using System.Collections.Generic;
using System.Text;
using Xunit;
using OpenCvSharp.Cuda;

namespace OpenCvSharp.Tests.cuda.GpuMat;

public class GpuMatTest : CudaTestBase
{
    [Fact]
    public void GpuMatUploadAndDownloadTest()
    {
        EnsureCuda();

        using var cpuSrc = new Mat(100, 100, MatType.CV_8UC1, new Scalar(128));

        using var gpuMat = new OpenCvSharp.Cuda.GpuMat();
        gpuMat.Upload(cpuSrc);

        Assert.False(gpuMat.Empty());
        Assert.Equal(cpuSrc.Rows, gpuMat.Rows);
        Assert.Equal(cpuSrc.Cols, gpuMat.Cols);

        using var cpuDst = new Mat();
        gpuMat.Download(cpuDst);

        ImageEquals(cpuSrc, cpuDst);
    }

    [Fact]
    public void GpuMatSimpleArithmeticTest()
    {
        EnsureCuda();

        using var cpuSrc = new Mat(50, 50, MatType.CV_8UC1, new Scalar(10));
        using var gpuSrc = new OpenCvSharp.Cuda.GpuMat(cpuSrc);
        using var gpuDst = new OpenCvSharp.Cuda.GpuMat();

        Cv2.Add(gpuSrc, new Scalar(50), gpuDst);

        using var cpuDst = new Mat();
        gpuDst.Download(cpuDst);

        // The result should be 60 (10 + 50)
        double min, max;
        cpuDst.MinMaxLoc(out min, out max);
        Assert.Equal(60, min);
        Assert.Equal(60, max);
    }
}
