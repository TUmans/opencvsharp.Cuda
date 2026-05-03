using System;
using Xunit;
using OpenCvSharp;
using OpenCvSharp.Cuda;
using OpenCvSharp.Tests.Cuda;

namespace OpenCvSharp.Tests.Cuda;

public class CudaCannyEdgeDetectorTest : CudaTestBase
{
    [Fact]
    public void CannyEdgeDetector_PropertiesTest()
    {
        VerifyCudaSupport();

        // 1. Arrange: Create the detector with initial values
        using var canny = OpenCvSharp.Cuda.CannyEdgeDetector.Create(lowThresh: 50.0, highThresh: 100.0, appertureSize: 3, l2Gradient: false);

        // 2. Assert initial values
        Assert.Equal(50.0, canny.LowThreshold);
        Assert.Equal(100.0, canny.HighThreshold);
        Assert.Equal(3, canny.AppertureSize);
        Assert.False(canny.L2Gradient);

        // 3. Act: Modify the properties
        canny.LowThreshold = 30.0;
        canny.HighThreshold = 120.0;
        canny.AppertureSize = 5; // Must be 3, 5, or 7
        canny.L2Gradient = true;

        // 4. Assert modified values
        Assert.Equal(30.0, canny.LowThreshold);
        Assert.Equal(120.0, canny.HighThreshold);
        Assert.Equal(5, canny.AppertureSize);
        Assert.True(canny.L2Gradient);
    }
}
