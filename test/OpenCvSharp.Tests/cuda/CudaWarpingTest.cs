using System;
using System.Collections.Generic;
using System.Text;
using Xunit;
using OpenCvSharp.Cuda;

namespace OpenCvSharp.Tests.Cuda;

public class CudaWarpingTest : CudaTestBase
{
    [Fact]
    public void BuildWarpAffineMapsTest()
    {
        VerifyCudaSupport();

        // Arrange: 2x3 Transformation Matrix for Affine
        double[] mData = { 1.0, 0.0, 10.0,   // shift right by 10
                               0.0, 1.0, 5.0 };  // shift down by 5
        using var cpuM = Mat.FromPixelData(2, 3, MatType.CV_64FC1, mData);
        Size dsize = new Size(100, 100);

        // Act
        var gpuXMap = new GpuMat();
        var gpuYMap = new GpuMat();
        Cv2.Cuda.BuildWarpAffineMaps(cpuM, false, dsize, gpuXMap, gpuYMap);

        // Assert
        Assert.False(gpuXMap.Empty());
        Assert.False(gpuYMap.Empty());
        Assert.Equal(100, gpuXMap.Cols);
        Assert.Equal(100, gpuYMap.Rows);

        // Cleanup
        gpuXMap.Dispose();
        gpuYMap.Dispose();
    }

    [Fact]
    public void BuildWarpPerspectiveMapsTest()
    {
        VerifyCudaSupport();

        // Arrange: 3x3 Transformation Matrix for Perspective
        double[] mData = { 1.0, 0.0, 10.0,
                               0.0, 1.0, 5.0,
                               0.0, 0.0, 1.0 };
        using var cpuM = Mat.FromPixelData( 3, 3, MatType.CV_64FC1, mData);
        Size dsize = new Size(100, 100);

        // Act
        var gpuXMap = new GpuMat();
        var gpuYMap = new GpuMat();
        Cv2.Cuda.BuildWarpPerspectiveMaps(cpuM, false, dsize, gpuXMap, gpuYMap);

        // Assert
        Assert.False(gpuXMap.Empty());
        Assert.False(gpuYMap.Empty());

        gpuXMap.Dispose();
        gpuYMap.Dispose();
    }
}
