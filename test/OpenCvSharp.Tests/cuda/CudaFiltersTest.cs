using System;
using System.Collections.Generic;
using System.Text;
using OpenCvSharp.Cuda;
using OpenCvSharp.Tests;
using Xunit;

namespace OpenCvSharp.Tests.Cuda;

public class CudaFiltersTest : CudaTestBase
{
    [Fact]
    public void CreateBoxFilterTest()
    {
        VerifyCudaSupport();

        // Arrange: 5x5 Matrix of zeros, with a single '90' in the very center.
        using var cpuSrc = new Mat(5, 5, MatType.CV_8UC1, new Scalar(0));
        cpuSrc.Set<byte>(2, 2, 90);

        using var gpuSrc = new GpuMat(); gpuSrc.Upload(cpuSrc);
        using var gpuDst = new GpuMat();

        // Act: 3x3 average filter
        using var filter = OpenCvSharp.Cuda.Filter.CreateBoxFilter(MatType.CV_8UC1, MatType.CV_8UC1, new Size(3, 3));
        filter.Apply(gpuSrc, gpuDst);

        // Assert
        using var cpuDst = new Mat();
        gpuDst.Download(cpuDst);

        // A 3x3 box average of a single '90' pixel surrounded by zeros = 90 / 9 = 10.
        // The center 3x3 grid should now all be 10.
        Assert.Equal(10, cpuDst.At<byte>(2, 2)); // Center
        Assert.Equal(10, cpuDst.At<byte>(1, 1)); // Top Left of the 3x3 area

        // Pixels outside the 3x3 reach should still be 0
        Assert.Equal(0, cpuDst.At<byte>(0, 0));
    }

    [Fact]
    public void CreateBoxMaxFilter()
    {
        VerifyCudaSupport();

        // Arrange: 5x5 Matrix of zeros, with a single '100' in the center.
        using var cpuSrc = new Mat(5, 5, MatType.CV_8UC1, new Scalar(0));
        cpuSrc.Set<byte>(2, 2, 100);

        using var gpuSrc = new GpuMat(); gpuSrc.Upload(cpuSrc);
        using var gpuDst = new GpuMat();

        // Act: 3x3 Max filter
        using var filter = OpenCvSharp.Cuda.Filter.CreateBoxMaxFilter(MatType.CV_8UC1, new Size(3, 3));
        filter.Apply(gpuSrc, gpuDst);

        // Assert
        using var cpuDst = new Mat();
        gpuDst.Download(cpuDst);

        // The max filter expands the brightest pixel. The center '100' should expand to a 3x3 square.
        Assert.Equal(100, cpuDst.At<byte>(2, 2)); // Center
        Assert.Equal(100, cpuDst.At<byte>(1, 1)); // Top Left of the 3x3 area

        // Pixels outside the 3x3 reach should still be 0
        Assert.Equal(0, cpuDst.At<byte>(0, 0));
    }

    [Fact]
    public void CreateBoxMinFilter()
    {
        VerifyCudaSupport();

        // Arrange: 5x5 Matrix of 255s, with a single '0' in the center.
        using var cpuSrc = new Mat(5, 5, MatType.CV_8UC1, new Scalar(255));
        cpuSrc.Set<byte>(2, 2, 0);

        using var gpuSrc = new GpuMat(); gpuSrc.Upload(cpuSrc);
        using var gpuDst = new GpuMat();

        // Act: 3x3 Min filter
        using var filter = OpenCvSharp.Cuda.Filter.CreateBoxMinFilter(MatType.CV_8UC1, new Size(3, 3));
        filter.Apply(gpuSrc, gpuDst);

        // Assert
        using var cpuDst = new Mat();
        gpuDst.Download(cpuDst);

        // The min filter expands the darkest pixel. The center '0' should expand to a 3x3 square.
        Assert.Equal(0, cpuDst.At<byte>(2, 2)); // Center
        Assert.Equal(0, cpuDst.At<byte>(1, 1)); // Top Left of the 3x3 area

        // Pixels outside the 3x3 reach should still be 255
        Assert.Equal(255, cpuDst.At<byte>(0, 0));
    }

    [Fact]
    public void CreateColumnSumFilter()
    {
        VerifyCudaSupport();

        // Arrange: 3x3 Matrix filled with 1s
        using var cpuSrc = new Mat(3, 3, MatType.CV_8UC1, new Scalar(1));
        using var gpuSrc = new GpuMat(); gpuSrc.Upload(cpuSrc);
        using var gpuDst = new GpuMat();

        // Act: Create a vertical filter of size 3. 
        // We output to CV_32SC1 (32-bit Int) to handle the sum results.
        using var filter = OpenCvSharp.Cuda.Filter.CreateColumnSumFilter(
            MatType.CV_8UC1,
            MatType.CV_32FC1,
            ksize: 3);

        filter.Apply(gpuSrc, gpuDst);

        // Assert
        using var cpuDst = new Mat();
        gpuDst.Download(cpuDst);

        Assert.False(cpuDst.Empty());
        Assert.Equal(MatType.CV_32FC1, cpuDst.Type());

        float resultValue = cpuDst.At<float>(1, 1);
        Assert.InRange(resultValue, 2.9f, 3.1f);
    }

    [Fact]
    public void CreateDerivFilter()
    {
        VerifyCudaSupport();

        // Arrange: Create a 10x10 gradient image
        // Row 0: [0, 10, 20, 30, 40, ...]
        using var cpuSrc = new Mat(10, 10, MatType.CV_8UC1);
        for (int x = 0; x < 10; x++)
        {
            using var col = cpuSrc.Col(x);
            col.SetTo(new Scalar(x * 10));
        }

        using var gpuSrc = new GpuMat(); gpuSrc.Upload(cpuSrc);
        using var gpuDst = new GpuMat();

        // Act: First derivative in X direction (dx=1, dy=0)
        // Aperture size 3, normalized.
        // Using CV_32F output to handle precision.
        using var filter = OpenCvSharp.Cuda.Filter.CreateDerivFilter(
            MatType.CV_8UC1,
            MatType.CV_32FC1,
            dx: 1, dy: 0, ksize: 3,
            normalize: true);

        filter.Apply(gpuSrc, gpuDst);

        // Assert
        using var cpuDst = new Mat();
        gpuDst.Download(cpuDst);

        Assert.False(cpuDst.Empty());

        // Because the gradient is 10 units per pixel, 
        // the first derivative (slope) should be 10.
        // We check a center pixel (to avoid border effects).
        float derivativeValue = cpuDst.At<float>(5, 5);

        // Allow small float tolerance
        Assert.InRange(derivativeValue, 9.9f, 10.1f);
    }
}
