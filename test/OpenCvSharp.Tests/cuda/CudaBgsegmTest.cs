using System;
using System.Collections.Generic;
using System.Text;
using OpenCvSharp.Cuda;
using Xunit;

namespace OpenCvSharp.Tests.Cuda;

public class CudaBgsegmTest : CudaTestBase
{
    [Fact]
    public void BackgroundSubtractorMOG2()
    {
        VerifyCudaSupport();

        try
        {
            using var mog2 = OpenCvSharp.Cuda.BackgroundSubtractorMOG2.Create(detectShadows: false);
            using var cpuFrame = new Mat(100, 100, MatType.CV_8UC1, new Scalar(0));
            using var gpuFrame = new GpuMat();
            using var gpuFgMask = new GpuMat();

            // Frame 1: Train background
            gpuFrame.Upload(cpuFrame);
            mog2.Apply(gpuFrame, gpuFgMask, learningRate: 1.0);

            // Frame 2: Introduce foreground
            Cv2.Rectangle(cpuFrame, new Rect(40, 40, 20, 20), new Scalar(255), -1);
            gpuFrame.Upload(cpuFrame);
            mog2.Apply(gpuFrame, gpuFgMask, learningRate: 0.0);

            // Assert
            using var cpuFgMask = new Mat();
            gpuFgMask.Download(cpuFgMask);

            Assert.False(cpuFgMask.Empty());
            Assert.Equal(0, cpuFgMask.At<byte>(10, 10)); // Background
            Assert.Equal(255, cpuFgMask.At<byte>(50, 50)); // Foreground
        }
        catch (OpenCVException ex) when (ex.Message.Contains("disabled") || ex.Message.Contains("Not Implemented"))
        {
            Assert.Skip("The called functionality is disabled for current build or platform");
        }
    }

    [Fact]
    public void BackgroundSubtractorMOG()
    {
        VerifyCudaSupport();

        try
        {
            using var mog = OpenCvSharp.Cuda.BackgroundSubtractorMOG.Create();
            using var cpuFrame = new Mat(100, 100, MatType.CV_8UC1, new Scalar(0));
            using var gpuFrame = new GpuMat();
            using var gpuFgMask = new GpuMat();

            // Frame 1: Train background
            gpuFrame.Upload(cpuFrame);
            mog.Apply(gpuFrame, gpuFgMask, learningRate: 1.0);

            // Frame 2: Introduce foreground
            Cv2.Rectangle(cpuFrame, new Rect(40, 40, 20, 20), new Scalar(255), -1);
            gpuFrame.Upload(cpuFrame);
            mog.Apply(gpuFrame, gpuFgMask, learningRate: 0.0);

            // Assert
            using var cpuFgMask = new Mat();
            gpuFgMask.Download(cpuFgMask);

            Assert.False(cpuFgMask.Empty());
            Assert.Equal(0, cpuFgMask.At<byte>(10, 10)); // Background
            Assert.Equal(255, cpuFgMask.At<byte>(50, 50)); // Foreground
        }
        catch (OpenCVException ex) when (ex.Message.Contains("disabled") || ex.Message.Contains("Not Implemented"))
        {
            Assert.Skip("The called functionality is disabled for current build or platform");
        }
    }

    [Fact]
    public void MOG_PropertiesAndMethodsTest()
    {
        VerifyCudaSupport();

        try
        {
            // 1. Create Subtractor
            using var mog = OpenCvSharp.Cuda.BackgroundSubtractorMOG.Create();

            // 2. Test Setters and Getters
            mog.History = 300;
            Assert.Equal(300, mog.History);

            mog.NMixtures = 6;
            Assert.Equal(6, mog.NMixtures);

            mog.BackgroundRatio = 0.5;
            Assert.Equal(0.5, mog.BackgroundRatio);

            mog.NoiseSigma = 0.1;
            Assert.InRange(mog.NoiseSigma,0.08,0.11 );

            // 3. Test Apply with Known Foreground Mask
            using var cpuFrame = new Mat(100, 100, MatType.CV_8UC1, new Scalar(100));
            using var gpuFrame = new GpuMat(); gpuFrame.Upload(cpuFrame);

            using var knownFgCpu = new Mat(100, 100, MatType.CV_8UC1, new Scalar(0));
            Cv2.Rectangle(knownFgCpu, new Rect(10, 10, 20, 20), new Scalar(255), -1);
            using var knownFgGpu = new GpuMat(); knownFgGpu.Upload(knownFgCpu);

            using var gpuFgMask = new GpuMat();

            // Apply
            mog.Apply(gpuFrame, knownFgGpu, gpuFgMask, learningRate: 0.1);
            Assert.False(gpuFgMask.Empty());

            // 4. Test GetBackgroundImage
            using var gpuBgImage = new GpuMat();
            mog.GetBackgroundImage(gpuBgImage);
            Assert.False(gpuBgImage.Empty());
        }
        catch (OpenCVException ex) when (ex.Message.Contains("disabled") || ex.Message.Contains("Not Implemented"))
        {
            Assert.Skip("The called functionality is disabled for current build or platform");
        }
    }

    [Fact]
    public void MOG2_InheritedPropertiesTest()
    {
        // Skip if no CUDA device is available
        VerifyCudaSupport();

        try
        {
            // 1. Arrange: Create the CUDA version of MOG2
            using var mog2 = OpenCvSharp.Cuda.BackgroundSubtractorMOG2.Create();

            // 2. Act & Assert: Test every inherited property

            // Background Ratio
            mog2.BackgroundRatio = 0.85;
            AssertAround(mog2.BackgroundRatio, 0.85, 0.01);

            // Complexity Reduction Threshold
            mog2.ComplexityReductionThreshold = 0.04;
            AssertAround(mog2.ComplexityReductionThreshold, 0.04, 0.01);

            // Detect Shadows
            mog2.DetectShadows = false;
            Assert.False(mog2.DetectShadows);
            mog2.DetectShadows = true;
            Assert.True(mog2.DetectShadows);

            // History
            mog2.History = 300;
            AssertAround(mog2.History, 300, 0.01);
            
            // NMixtures
            mog2.NMixtures = 6;
            AssertAround(mog2.NMixtures, 6, 0.01);

            // Shadow Threshold
            mog2.ShadowThreshold = 0.6;
            AssertAround(mog2.ShadowThreshold, 0.6, 0.01);

            // Shadow Value
            mog2.ShadowValue = 100;
            AssertAround(mog2.ShadowValue, 100, 0.01);

            // Variance Initialization
            mog2.VarInit = 16.0;
            AssertAround(mog2.VarInit, 16, 0.01);

            // Variance Maximum
            mog2.VarMax = 100.0;
            AssertAround(mog2.VarMax, 100, 0.01);

            // Variance Minimum
            mog2.VarMin = 5.0;
            AssertAround(mog2.VarMin, 5.0, 0.01);

            // Variance Threshold
            mog2.VarThreshold = 20.0;
            AssertAround(mog2.VarThreshold, 20.0, 0.01);

            // Variance Threshold Gen
            mog2.VarThresholdGen = 10.0;
            AssertAround(mog2.VarThresholdGen, 10.0, 0.01);
        }
        catch (OpenCVException ex) when (ex.Message.Contains("disabled") || ex.Message.Contains("Not Implemented"))
        {
            // Graceful exit for environments where cudabgsegm is missing
            return;
        }
    }

    [Fact]
    public void MOG2_InheritedApplyTest()
    {
        VerifyCudaSupport();

        // This tests that calling the BASE class Apply method with a GpuMat 
        // correctly routes to the GPU and doesn't crash expecting a CPU Mat.
        using var mog2 = OpenCvSharp.Cuda.BackgroundSubtractorMOG2.Create();

        using var cpuFrame = new Mat(100, 100, MatType.CV_8UC1, new Scalar(128));
        using var gpuFrame = new GpuMat();
        gpuFrame.Upload(cpuFrame);

        using var gpuFgMask = new GpuMat();

        // Act: Call the standard Apply method (inherited from CPU version)
        // but pass GpuMat objects (which implicitly cast to InputArray/OutputArray)
        var exception = Record.Exception(() =>
        {
            mog2.Apply(gpuFrame, gpuFgMask, learningRate: 0.1);
        });

        // Assert
        Assert.Null(exception);
        Assert.False(gpuFgMask.Empty());
    }
}


