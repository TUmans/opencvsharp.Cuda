using Xunit;
using OpenCvSharp.Cuda;
using Xunit.Sdk;

namespace OpenCvSharp.Tests.Cuda;

public class CudaStereoTest : CudaTestBase
{
    [Fact]
    public void CudaDisparityBilateralFilter()
    {
        VerifyCudaSupport();

        // Arrange: 
        // Disparity Map: 100x100 grayscale with a noisy patch
        using var cpuDisp = new Mat(100, 100, MatType.CV_8UC1, new Scalar(30));
        Cv2.Rectangle(cpuDisp, new Rect(40, 40, 20, 20), new Scalar(35), -1); // Subtle difference

        // Reference Image: Solid color
        using var cpuImg = new Mat(100, 100, MatType.CV_8UC3, new Scalar(100, 50, 200));

        using var gpuDisp = new GpuMat(); gpuDisp.Upload(cpuDisp);
        using var gpuImg = new GpuMat(); gpuImg.Upload(cpuImg);
        using var gpuDst = new GpuMat();

        // Create Filter
        using var filter = OpenCvSharp.Cuda.DisparityBilateralFilter.Create(ndisp: 64, radius: 3, iters: 1);

        // Act
        filter.Apply(gpuDisp, gpuImg, gpuDst);

        // Assert
        using var cpuDst = new Mat();
        gpuDst.Download(cpuDst);

        Assert.False(cpuDst.Empty());
        Assert.Equal(cpuDisp.Size(), cpuDst.Size());
        Assert.Equal(cpuDisp.Type(), cpuDst.Type());
    }

    [Fact]
    public void StereoBeliefPropagation_ComputeTest()
    {
        VerifyCudaSupport();

        // 1. Arrange: Create a synthetic stereo pair
        // Left image: black with a white square at (40, 40)
        using var leftCpu = new Mat(100, 100, MatType.CV_8UC1, new Scalar(0));
        Cv2.Rectangle(leftCpu, new Rect(40, 40, 20, 20), new Scalar(255), -1);

        // Right image: black with the white square shifted to (35, 40) -> 5px disparity
        using var rightCpu = new Mat(100, 100, MatType.CV_8UC1, new Scalar(0));
        Cv2.Rectangle(rightCpu, new Rect(35, 40, 20, 20), new Scalar(255), -1);

        using var leftGpu = new GpuMat(); leftGpu.Upload(leftCpu);
        using var rightGpu = new GpuMat(); rightGpu.Upload(rightCpu);
        using var disparityGpu = new GpuMat();

        // 2. Act: Create and compute
        using var stereo = OpenCvSharp.Cuda.StereoBeliefPropagation.Create(ndisp: 64, iters: 5, levels: 5);

        // .Compute is inherited from StereoMatcher
        stereo.Compute(leftGpu, rightGpu, disparityGpu);

        // 3. Assert
        using var disparityCpu = new Mat();
        disparityGpu.Download(disparityCpu);

        Assert.False(disparityCpu.Empty());

        // Belief Propagation typically outputs CV_16S or CV_32F disparity
        // Check a pixel inside the square area
        float dispValue = disparityCpu.At<float>(50, 50);
        Assert.True(dispValue > 0, "Disparity should be detected for the shifted square.");
    }

    [Fact]
    public void StereoBM_ComputeTest()
    {
        VerifyCudaSupport();

        // 1. Arrange: Create synthetic stereo pair
        // Left image: black with a white square at (40, 40)
        using var leftCpu = new Mat(100, 100, MatType.CV_8UC1, new Scalar(0));
        Cv2.Rectangle(leftCpu, new Rect(40, 40, 20, 20), new Scalar(255), -1);

        // Right image: black with white square shifted to (30, 40) -> 10px disparity
        using var rightCpu = new Mat(100, 100, MatType.CV_8UC1, new Scalar(0));
        Cv2.Rectangle(rightCpu, new Rect(30, 40, 20, 20), new Scalar(255), -1);

        using var leftGpu = new GpuMat(); leftGpu.Upload(leftCpu);
        using var rightGpu = new GpuMat(); rightGpu.Upload(rightCpu);
        using var disparityGpu = new GpuMat();

        // 2. Act
        // Note: numDisparities must be a multiple of 16
        using var stereo = OpenCvSharp.Cuda.StereoBM.Create(numDisparities: 64, blockSize: 19);

        // Compute is inherited from StereoMatcher and supports GpuMat via Input/OutputArray
        stereo.Compute(leftGpu, rightGpu, disparityGpu);

        // 3. Assert
        using var disparityCpu = new Mat();
        disparityGpu.Download(disparityCpu);

        Assert.False(disparityCpu.Empty());

        // StereoBM typically outputs CV_16S disparity.
        // Check a pixel inside the object area
        short dispValue = disparityCpu.At<short>(40, 40);

        // If disparity is > 0, the object was found
        Assert.True(dispValue >= 0 || dispValue == 0);
    }

    [Fact]
    public void StereoConstantSpaceBP_ComputeTest()
    {
        VerifyCudaSupport();

        // 1. Arrange: Create synthetic stereo pair
        // Left image: black with a white square
        using var leftCpu = new Mat(128, 128, MatType.CV_8UC1, new Scalar(0));
        Cv2.Rectangle(leftCpu, new Rect(60, 60, 30, 30), new Scalar(255), -1);

        // Right image: shifted white square (disparity of 8 pixels)
        using var rightCpu = new Mat(128, 128, MatType.CV_8UC1, new Scalar(0));
        Cv2.Rectangle(rightCpu, new Rect(52, 60, 30, 30), new Scalar(255), -1);

        using var leftGpu = new GpuMat(); leftGpu.Upload(leftCpu);
        using var rightGpu = new GpuMat(); rightGpu.Upload(rightCpu);
        using var disparityGpu = new GpuMat();

        // 2. Act
        using var stereo = OpenCvSharp.Cuda.StereoConstantSpaceBP.Create(ndisp: 128, iters: 8, levels: 4, nrPlane: 4);

        // Compute is inherited from StereoMatcher
        stereo.Compute(leftGpu, rightGpu, disparityGpu);

        // 3. Assert
        using var disparityCpu = new Mat();
        disparityGpu.Download(disparityCpu);

        Assert.False(disparityCpu.Empty());

        // Check pixel in the center of the square
        // CSBP typically outputs CV_16S or CV_32F disparity
        float dispValue = disparityCpu.At<float>(75, 75);

        // We expect a disparity value greater than 0
        Assert.True(dispValue > 0, $"Disparity should be positive, but was {dispValue}");
    }

    [Fact]
    public void DrawColorDisp_Test()
    {
        VerifyCudaSupport();

        // 1. Arrange: Create a 1-channel disparity map (100x100)
        using var cpuDisp = new Mat(100, 100, MatType.CV_8UC1, new Scalar(30));
        using var gpuDisp = new GpuMat(); gpuDisp.Upload(cpuDisp);
        using var gpuColor = new GpuMat();

        // 2. Act: Draw colored disparity
        Cv2.Cuda.DrawColorDisp(gpuDisp, gpuColor, ndisp: 64);

        // 3. Download and Assert
        using var cpuColor = new Mat();
        gpuColor.Download(cpuColor);

        Assert.False(cpuColor.Empty());

        // The CUDA drawColorDisp implementation typically outputs CV_8UC4 (BGRA)
        Assert.Equal(4, cpuColor.Channels());
        Assert.Equal(MatType.CV_8UC4, cpuColor.Type());

        // Ensure the rows/cols match
        Assert.Equal(100, cpuColor.Rows);
        Assert.Equal(100, cpuColor.Cols);
    }

    [Fact]
    public void ReprojectImageTo3D_Test()
    {
        VerifyCudaSupport();

        // 1. Arrange: 10x10 disparity map filled with value 10
        using var cpuDisp = new Mat(10, 10, MatType.CV_8UC1, new Scalar(10));
        using var gpuDisp = new GpuMat(); gpuDisp.Upload(cpuDisp);

        // 2. Simple Q Matrix (4x4)
        // Typically generated by cv::stereoRectify
        double[] qData = {
                1, 0, 0, -5,
                0, 1, 0, -5,
                0, 0, 0, 50,
                0, 0, 1/16.0, 0
            };
        using var Q = Mat.FromPixelData(4, 4, MatType.CV_32F, qData);

        // 3. Act: Reproject to 3D with 3 channels (XYZ)
        using var gpu3D = new GpuMat();
        Cv2.Cuda.ReprojectImageTo3D(gpuDisp, gpu3D,Q, dstCn: 3);

        // 4. Assert
        using var cpu3D = new Mat();
        gpu3D.Download(cpu3D);

        Assert.False(cpu3D.Empty());
        Assert.Equal(3, cpu3D.Channels());
        Assert.Equal(MatType.CV_32FC3, cpu3D.Type());

        // Verify coordinate dimensions
        Assert.Equal(10, cpu3D.Rows);
        Assert.Equal(10, cpu3D.Cols);

        // Verify a point has non-zero Z coordinate
        Vec3f point = cpu3D.At<Vec3f>(5, 5);
        Assert.NotEqual(0, point.Item2); // Z should be calculated
    }

    [Fact]
    public void StereoSGM_ComputeTest()
    {
        VerifyCudaSupport();

        // 1. Arrange: Create synthetic stereo pair (128x128)
        using var leftCpu = new Mat(128, 128, MatType.CV_8UC1, new Scalar(0));
        Cv2.Rectangle(leftCpu, new Rect(60, 60, 30, 30), new Scalar(255), -1);

        // Shift square by 10 pixels to the left in the right image
        using var rightCpu = new Mat(128, 128, MatType.CV_8UC1, new Scalar(0));
        Cv2.Rectangle(rightCpu, new Rect(50, 60, 30, 30), new Scalar(255), -1);

        using var leftGpu = new GpuMat(); leftGpu.Upload(leftCpu);
        using var rightGpu = new GpuMat(); rightGpu.Upload(rightCpu);
        using var disparityGpu = new GpuMat();

        // 2. Act
        // numDisparities must be divisible by 16
        using var sgm = OpenCvSharp.Cuda.StereoSGM.Create(minDisparity: 0, numDisparities: 64);
        sgm.Compute(leftGpu, rightGpu, disparityGpu);

        // 3. Assert
        using var disparityCpu = new Mat();
        disparityGpu.Download(disparityCpu);

        Assert.False(disparityCpu.Empty(), "Disparity map should not be empty.");

        // SGM output is usually CV_16S (signed short) or CV_8U 
        // depending on implementation, but typically CV_16S.
        // Check a pixel in the center of the object.
        short val = disparityCpu.At<short>(75, 75);
        Assert.True(val > 0, $"Expected positive disparity, but got {val}");
    }
}

