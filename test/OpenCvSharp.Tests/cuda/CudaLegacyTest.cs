using Xunit;
using OpenCvSharp.Cuda;
using Xunit.Sdk;

namespace OpenCvSharp.Tests.Cuda;


public class CudaLegacyTest : CudaTestBase
{
    [Fact]
    public void CalcOpticalFlowBMTest()
    {
        VerifyCudaSupport();

        // Arrange
        using var cpuPrev = new Mat(64, 64, MatType.CV_8UC1, new Scalar(0));
        using var cpuCurr = new Mat(64, 64, MatType.CV_8UC1, new Scalar(0));

        // Create a moving square (+2, +2)
        Cv2.Rectangle(cpuPrev, new Rect(10, 10, 15, 15), new Scalar(255), -1);
        Cv2.Rectangle(cpuCurr, new Rect(12, 12, 15, 15), new Scalar(255), -1);

        using var gpuPrev = new GpuMat(); gpuPrev.Upload(cpuPrev);
        using var gpuCurr = new GpuMat(); gpuCurr.Upload(cpuCurr);

        using var velx = new GpuMat();
        using var vely = new GpuMat();
        using var buf = new GpuMat();

        // Act
        Cv2.Cuda.CalcOpticalFlowBM(
            gpuPrev, gpuCurr,
            blockSize: new Size(15, 15),
            shiftSize: new Size(1, 1),
            maxRange: new Size(15, 15),
            usePrevious: false,
            velx, vely, buf);

        // Basic assertions
        Assert.False(velx.Empty(), "X velocity matrix should not be empty.");
        Assert.False(vely.Empty(), "Y velocity matrix should not be empty.");

        // Download results
        using var cpuVelX = new Mat();
        using var cpuVelY = new Mat();
        velx.Download(cpuVelX);
        vely.Download(cpuVelY);

        Assert.Equal(cpuVelX.Size(), cpuVelY.Size());

        // ---- Detect scaling (BM often uses fixed-point, e.g. *16) ----
        double rawMeanX = Cv2.Mean(cpuVelX).Val0;
        double rawMeanY = Cv2.Mean(cpuVelY).Val0;

        double scale = 1.0;

        // Heuristic: if values are large, assume fixed-point scaling
        if (Math.Abs(rawMeanX) > 10 || Math.Abs(rawMeanY) > 10)
            scale = 16.0;

        // ---- Focus on moving region ----
        var roi = new Rect(12, 12, 10, 10); // inside moved square
        using var subX = new Mat(cpuVelX, roi);
        using var subY = new Mat(cpuVelY, roi);

        double avgX = Cv2.Mean(subX).Val0 / scale;
        double avgY = Cv2.Mean(subY).Val0 / scale;

        // ---- Validate motion (should be approx +2, +2) ----
        Assert.InRange(Math.Abs(avgX), 1.5, 2.5);
        Assert.InRange(Math.Abs(avgY), 1.5, 2.5);

        // ---- Validate background is ~0 ----
        var bg = new Rect(0, 0, 10, 10);
        using var bgX = new Mat(cpuVelX, bg);
        using var bgY = new Mat(cpuVelY, bg);

        double bgAvgX = Cv2.Mean(bgX).Val0 / scale;
        double bgAvgY = Cv2.Mean(bgY).Val0 / scale;

        Assert.InRange(bgAvgX, -2.5, 2.5);
        Assert.InRange(bgAvgY, -2.5, 2.5);
    }


    [Fact]
    public void ConnectivityMaskTest()
    {
        VerifyCudaSupport();

        // Arrange
        // Create a 5x5 image.
        using var cpuImg = new Mat(5, 5, MatType.CV_8UC1, new Scalar(50));
        // Add a "connected" shape with intensity 100 in the middle
        cpuImg.Set<byte>(2, 2, 100);
        cpuImg.Set<byte>(2, 3, 105);

        using var gpuImg = new GpuMat(); gpuImg.Upload(cpuImg);
        using var gpuMask = new GpuMat();

        // Act: Find pixels between 90 and 110
        try
        {
            // Act
            Cv2.Cuda.ConnectivityMask(gpuImg, gpuMask, new Scalar(90), new Scalar(110));
        }
        catch (OpenCVException ex) when (ex.Message.Contains("disabled for current build"))
        {
            // The cudalegacy module is often disabled in modern OpenCV binaries.
            // If it's disabled, gracefully exit the test without failing it.
            Assert.Skip("The called functionality is disabled for current build or platform");
        }


        // Assert
        using var cpuMask = new Mat();
        gpuMask.Download(cpuMask);

        Assert.False(cpuMask.Empty());
        Assert.Equal(MatType.CV_8UC1, cpuMask.Type());

        // The background (50) should be 0 in the mask
        Assert.Equal(0, cpuMask.At<byte>(0, 0));
        // The pixels within the lo/hi range should be 255
        Assert.Equal(255, cpuMask.At<byte>(2, 2));
        Assert.Equal(255, cpuMask.At<byte>(2, 3));
    }

    [Fact]
    public void BackgroundSubtractorGMG()
    {
        VerifyCudaSupport();

        try
        {
            // 1. Create GMG with exactly 1 initialization frame so it activates immediately.
            using var gmg = OpenCvSharp.Cuda.BackgroundSubtractorGMG.Create(initializationFrames: 1);

            using var cpuFrame = new Mat(100, 100, MatType.CV_8UC1, new Scalar(0));
            using var gpuFrame = new GpuMat();
            using var gpuFgMask = new GpuMat();

            // 2. FRAME 1: Pure black background (Initialization phase)
            gpuFrame.Upload(cpuFrame);

            // Using our Stream overload (stream = null) 30 because we need to train our model
            for (int i = 0; i < 30; i++)
            {
                gmg.Apply(gpuFrame, gpuFgMask, learningRate: 0.1);
            }

            // 3. FRAME 2: Introduce a moving object (white square)
            Cv2.Rectangle(cpuFrame, new Rect(40, 40, 20, 20), new Scalar(255), -1);
            gpuFrame.Upload(cpuFrame);

            // Apply frame 2 to extract the foreground
            gmg.Apply(gpuFrame, gpuFgMask, learningRate: 0.0);

            // 4. Download and Assert
            using var cpuFgMask = new Mat();
            gpuFgMask.Download(cpuFgMask);

            Assert.False(cpuFgMask.Empty(), "Foreground mask should not be empty.");
            Assert.Equal(MatType.CV_8UC1, cpuFgMask.Type());

            // Assert that the static background pixel is dark (0)
            Assert.Equal(0, cpuFgMask.At<byte>(10, 10));

            // Assert that the moving white square pixel is bright (255)
            Assert.Equal(255, cpuFgMask.At<byte>(50, 50));
        }
        catch (OpenCVException ex) when (ex.Message.Contains("disabled") || ex.Message.Contains("Not Implemented"))
        {
            // Graceful exit: cudabgsegm is an extra module and not always compiled 
            // into all OpenCV distribution binaries.
            Assert.Skip("The called functionality is disabled for current build or platform");
        }
    }


    [Fact]
    public void BackgroundSubtractorFGD()
    {
        VerifyCudaSupport();

        try
        {
            using var fgd = OpenCvSharp.Cuda.BackgroundSubtractorFGD.Create();

            // FGD REQUIRES 3-channel images (BGR). It will crash if given 1-channel grayscale.
            using var cpuFrame = new Mat(100, 100, MatType.CV_8UC3, new Scalar(0, 0, 0));
            using var gpuFrame = new GpuMat();
            using var gpuFgMask = new GpuMat();

            // Frame 1: Train background
            gpuFrame.Upload(cpuFrame);
            fgd.Apply(gpuFrame, gpuFgMask, learningRate: 1.0);

            // Frame 2: Introduce foreground (A white square)
            Cv2.Rectangle(cpuFrame, new Rect(40, 40, 20, 20), new Scalar(255, 255, 255), -1);
            gpuFrame.Upload(cpuFrame);
            fgd.Apply(gpuFrame, gpuFgMask, learningRate: 0.0);

            // Assert
            using var cpuFgMask = new Mat();
            gpuFgMask.Download(cpuFgMask);

            Assert.False(cpuFgMask.Empty());
            // The output mask is always a 1-channel binary image (CV_8UC1)
            Assert.Equal(MatType.CV_8UC1, cpuFgMask.Type());

            // Background should be black (0)
            Assert.Equal(0, cpuFgMask.At<byte>(10, 10));
            // Foreground should be white (255)
            Assert.Equal(255, cpuFgMask.At<byte>(50, 50));
        }
        catch (OpenCVException ex) when (ex.Message.Contains("disabled") || ex.Message.Contains("Not Implemented"))
        {
            // Graceful exit if cudabgsegm is not compiled into the OpenCV binaries.
            Assert.Skip("The called functionality is disabled for current build or platform");
        }
    }

    [Fact]
    public void ImagePyramid_GetLayerTest()
    {
        VerifyCudaSupport();

        // 1. Arrange: 100x100 source image
        using var cpuSrc = new Mat(100, 100, MatType.CV_8UC1, new Scalar(128));
        using var gpuSrc = new GpuMat(); gpuSrc.Upload(cpuSrc);

        // 2. Act: Create pyramid
        using var pyramid = OpenCvSharp.Cuda.ImagePyramid.Create(gpuSrc, nLayers: 3);

        using var gpuLayer0 = new GpuMat();
        using var gpuLayer1 = new GpuMat();

        // Fetch the 100x100 layer
        pyramid.GetLayer(gpuLayer0, new Size(100, 100));
        // Fetch the 50x50 layer
        pyramid.GetLayer(gpuLayer1, new Size(50, 50));

        // 3. Assert
        Assert.Equal(100, gpuLayer0.Rows);
        Assert.Equal(50, gpuLayer1.Rows);
        Assert.False(gpuLayer1.Empty());
    }

    [Fact]
    public void CreateOpticalFlowNeedleMap_Test()
    {
        VerifyCudaSupport();

        // 1. Arrange: Create small flow components (u = horizontal, v = vertical)
        using var u = new GpuMat(20, 20, MatType.CV_32FC1, new Scalar(1.0f));
        using var v = new GpuMat(20, 20, MatType.CV_32FC1, new Scalar(1.0f));

        using var vertex = new GpuMat();
        using var colors = new GpuMat();

        // 2. Act
        Cv2.Cuda.CreateOpticalFlowNeedleMap(u, v, vertex, colors);

        Assert.False(vertex.Empty());
        Assert.False(colors.Empty());

        // Vertex is float vector field
        Assert.Equal(MatType.CV_32FC3, vertex.Type());

        // Colors may be float or 8-bit depending on backend
        Assert.True(
            colors.Type() == MatType.CV_32FC3 ||
            colors.Type() == MatType.CV_8UC4
        );
    }

    [Fact]
    public void Graphcut_Test()
    {
        VerifyCudaSupport();

        try
        {
            int rows = 10;
            int cols = 10;

            // terminals: 2-channel CV_32S (Source and Sink weights)
            using var terminals = new GpuMat(rows, cols, MatType.CV_32SC2, new Scalar(0, 0));
            // Neighborhood weights (all CV_32S)
            using var leftTransp = new GpuMat(cols, rows, MatType.CV_32SC1, new Scalar(1));
            using var rightTransp = new GpuMat(cols, rows, MatType.CV_32SC1, new Scalar(1));
            using var top = new GpuMat(rows, cols, MatType.CV_32SC1, new Scalar(1));
            using var bottom = new GpuMat(rows, cols, MatType.CV_32SC1, new Scalar(1));

            using var labels = new GpuMat(rows, cols, MatType.CV_8UC1);
            using var buf = new GpuMat();

            // Act
            Cv2.Cuda.Graphcut(terminals, leftTransp, rightTransp, top, bottom, labels, buf);

            // Assert
            Assert.False(labels.Empty());
            Assert.Equal(rows, labels.Rows);
            Assert.Equal(cols, labels.Cols);
        }
        catch (OpenCVException ex) when (ex.Message.Contains("disabled") || ex.Message.Contains("not implemented"))
        {
            Assert.Skip("The called functionality is disabled for current build or platform");
        }
    }

    [Fact]
    public void Graphcut8_Test()
    {
        VerifyCudaSupport();

        try
        {
            int rows = 10;
            int cols = 10;

            using var terminals = new GpuMat(rows, cols, MatType.CV_32SC2, new Scalar(0, 0));

            // Neighborhood weights (All 11 matrices!)
            using var leftT = new GpuMat(cols, rows, MatType.CV_32SC1, new Scalar(1));
            using var rightT = new GpuMat(cols, rows, MatType.CV_32SC1, new Scalar(1));
            using var top = new GpuMat(rows, cols, MatType.CV_32SC1, new Scalar(1));
            using var tLeft = new GpuMat(rows, cols, MatType.CV_32SC1, new Scalar(1));
            using var tRight = new GpuMat(rows, cols, MatType.CV_32SC1, new Scalar(1));
            using var bot = new GpuMat(rows, cols, MatType.CV_32SC1, new Scalar(1));
            using var bLeft = new GpuMat(rows, cols, MatType.CV_32SC1, new Scalar(1));
            using var bRight = new GpuMat(rows, cols, MatType.CV_32SC1, new Scalar(1));

            using var labels = new GpuMat(rows, cols, MatType.CV_8UC1);
            using var buf = new GpuMat();

            // Act
            Cv2.Cuda.Graphcut(terminals, leftT, rightT, top, tLeft, tRight, bot, bLeft, bRight, labels, buf);

            // Assert
            Assert.False(labels.Empty());
            Assert.Equal(rows, labels.Rows);
        }
        catch (OpenCVException ex) when (ex.Message.Contains("disabled") || ex.Message.Contains("not implemented"))
        {
            Assert.Skip("The called functionality is disabled for current build or platform");
        }
    }
}


