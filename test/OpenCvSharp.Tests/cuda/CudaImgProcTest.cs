using System;
using System.Collections.Generic;
using System.Text;
using OpenCvSharp.Cuda;
using OpenCvSharp.Modules.core.Enum;
using OpenCvSharp.Tests.Cuda;
using Xunit;

namespace OpenCvSharp.Tests.Cuda;

public class CudaImgProcTest : CudaTestBase
{
    [Fact]
    public void AlphaCompTest()
    {
        VerifyCudaSupport();

        // AlphaComp REQUIRES 4-channel images (RGBA/BGRA) (CV_8UC4 or CV_16UC4 or CV_32FC4)
        // Foreground: value 100 on first channel, 255 alpha (fully opaque)
        using var cpuImg1 = new Mat(2, 2, MatType.CV_8UC4, new Scalar(100, 0, 0, 255));
        // Background: value 50 on first channel, 255 alpha
        using var cpuImg2 = new Mat(2, 2, MatType.CV_8UC4, new Scalar(50, 0, 0, 255));

        using var gpuImg1 = new GpuMat();
        using var gpuImg2 = new GpuMat();
        gpuImg1.Upload(cpuImg1);
        gpuImg2.Upload(cpuImg2);

        // Act: Place Foreground OVER Background
        using var gpuDst = new GpuMat();
        Cv2.Cuda.AlphaComp(gpuImg1, gpuImg2, gpuDst, AlphaCompTypes.Over);

        // Download and Assert
        using var cpuDst = new Mat();
        gpuDst.Download(cpuDst);

        Assert.False(cpuDst.Empty());
        Assert.Equal(MatType.CV_8UC4, cpuDst.Type());

        // Because the foreground was fully opaque (alpha=255), ALPHA_OVER 
        // should result in the exact foreground image.
        Vec4b pixel = cpuDst.At<Vec4b>(0, 0);
        Assert.Equal(100, pixel.Item0); // First channel
        Assert.Equal(255, pixel.Item3); // Alpha channel
    }

    [Fact]
    public void BilateralFilterTest()
    {
        VerifyCudaSupport();

        // Bilateral Filter usually operates on CV_8UC1 or CV_8UC3
        // Create a completely flat, gray image.
        using var cpuSrc = new Mat(10, 10, MatType.CV_8UC1, new Scalar(128));
        using var gpuSrc = new GpuMat();
        gpuSrc.Upload(cpuSrc);

        // Act
        using var gpuDst = new GpuMat();
        Cv2.Cuda.BilateralFilter(
            gpuSrc, gpuDst,
            kernelSize: 5,
            sigmaColor: 50.0f,
            sigmaSpatial: 50.0f);

        // Download and Assert
        using var cpuDst = new Mat();
        gpuDst.Download(cpuDst);

        Assert.False(cpuDst.Empty());
        Assert.Equal(cpuSrc.Rows, cpuDst.Rows);
        Assert.Equal(cpuSrc.Cols, cpuDst.Cols);
        Assert.Equal(MatType.CV_8UC1, cpuDst.Type());

        // Since the input image was a solid color, filtering it should do absolutely 
        // nothing to the color. It should remain 128.
        Assert.Equal(128, cpuDst.At<byte>(5, 5));
    }

    [Fact]
    public void BlendLinearTest()
    {
        VerifyCudaSupport();

        // Arrange: 
        using var cpuImg1 = new Mat(5, 5, MatType.CV_8UC1, new Scalar(100));
        using var cpuImg2 = new Mat(5, 5, MatType.CV_8UC1, new Scalar(50));

        using var cpuW1 = new Mat(5, 5, MatType.CV_32FC1, new Scalar(0.8f));
        using var cpuW2 = new Mat(5, 5, MatType.CV_32FC1, new Scalar(0.2f));

        using var gpuImg1 = new GpuMat(); gpuImg1.Upload(cpuImg1);
        using var gpuImg2 = new GpuMat(); gpuImg2.Upload(cpuImg2);
        using var gpuW1 = new GpuMat(); gpuW1.Upload(cpuW1);
        using var gpuW2 = new GpuMat(); gpuW2.Upload(cpuW2);
        using var gpuDst = new GpuMat();

        // Act
        Cv2.Cuda.BlendLinear(gpuImg1, gpuImg2, gpuW1, gpuW2, gpuDst);

        // Assert
        using var cpuDst = new Mat();
        gpuDst.Download(cpuDst);

        Assert.False(cpuDst.Empty());

        // We expect ~90, but GPU Fused-Multiply-Add (FMA) might truncate it to 89.
        // We allow a tolerance of +/- 1.
        byte pixelValue = cpuDst.At<byte>(0, 0);
        Assert.InRange(pixelValue, 89, 91);
    }

    [Fact]
    public void CreateCannyEdgeDetector()
    {
        VerifyCudaSupport();

        // 1. Arrange
        // Create a 100x100 completely black image
        using var cpuSrc = new Mat(100, 100, MatType.CV_8UC1, new Scalar(0));

        // Make the left half completely white.
        // This creates a perfectly sharp vertical edge at column 50.
        Cv2.Rectangle(cpuSrc, new Rect(0, 0, 50, 100), new Scalar(255), -1);

        using var gpuSrc = new GpuMat(); gpuSrc.Upload(cpuSrc);
        using var gpuEdges = new GpuMat();

        // Create Canny detector with standard thresholds
        using var canny = OpenCvSharp.Cuda.CannyEdgeDetector.Create(lowThresh: 50.0, highThresh: 100.0);

        // 2. Act
        canny.Detect(gpuSrc, gpuEdges);

        // 3. Download and Assert
        using var cpuEdges = new Mat();
        gpuEdges.Download(cpuEdges);    

        Assert.False(cpuEdges.Empty());
        Assert.Equal(MatType.CV_8UC1, cpuEdges.Type());

        // A pixel far away from the edge should be 0 (No Edge)
        Assert.Equal(0, cpuEdges.At<byte>(50, 10));
        Assert.Equal(0, cpuEdges.At<byte>(50, 90));

        int edgeCount = 0;

        for (int y = 0; y < 100; y++)
        {
            for (int x = 48; x <= 52; x++)
            {
                if (cpuEdges.At<byte>(y, x) > 0)
                    edgeCount++;
            }
        }

        Assert.True(edgeCount > 50, "Expected a continuous vertical edge");
    }

    [Fact]
    public void CudaCLAHE()
    {
        VerifyCudaSupport();

        // 1. Arrange
        // Create a 100x100 image with terrible contrast.
        using var cpuSrc = new Mat(100, 100, MatType.CV_8UC1, new Scalar(100)); // Dark Gray

        for (int y = 0; y < 100; y++)
        {
            for (int x = 0; x < 100; x++)
            {
                cpuSrc.Set(y, x, (byte)(x)); // horizontal gradient 0–99
            }
        }

        // Draw a tiny square that is barely visibly lighter (150)
        Cv2.Rectangle(cpuSrc, new Rect(40, 40, 20, 20), new Scalar(150), -1);

        using var gpuSrc = new GpuMat(); gpuSrc.Upload(cpuSrc);
        using var gpuDst = new GpuMat();

        // Create CLAHE algorithm
        using var clahe = OpenCvSharp.Cuda.CLAHE.Create(clipLimit: 40.0, tileGridSize: new Size(8, 8));

        // 2. Act
        clahe.Apply(gpuSrc, gpuDst);

        // 3. Download and Assert
        using var cpuDst = new Mat();
        gpuDst.Download(cpuDst);

        Cv2.MinMaxLoc(cpuSrc, out double minBefore, out double maxBefore);
        Cv2.MinMaxLoc(cpuDst, out double minAfter, out double maxAfter);

        double rangeBefore = maxBefore - minBefore;
        double rangeAfter = maxAfter - minAfter;

        Assert.True(rangeAfter > rangeBefore,
            $"Expected expanded dynamic range, before={rangeBefore}, after={rangeAfter}");
    }

    [Fact]
    public void GoodFeaturesToTrack_DetectTest()
    {
        VerifyCudaSupport();

        // 1. Arrange: Create a black image with a white square
        using var cpuSrc = new Mat(100, 100, MatType.CV_8UC1, new Scalar(0));
        // Square from (30,30) to (70,70) -> 4 corners
        Cv2.Rectangle(cpuSrc, new Rect(30, 30, 40, 40), new Scalar(255), -1);

        using var gpuSrc = new GpuMat(); gpuSrc.Upload(cpuSrc);
        using var gpuCorners = new GpuMat();

        // 2. Act
        using var detector = OpenCvSharp.Cuda.CornersDetector.Create(
      MatType.CV_8UC1,
      maxCorners: 100,
      qualityLevel: 0.01,
      minDistance: 1,
      blockSize: 3
  );

        detector.Detect(gpuSrc, gpuCorners);

        // Important: check BEFORE download
        Assert.False(gpuCorners.Empty());

        using var cpuCorners = new Mat();
        gpuCorners.Download(cpuCorners);

        Assert.Equal(MatType.CV_32FC2, cpuCorners.Type());

        // Count check
        int count = cpuCorners.Rows * cpuCorners.Cols;
        Assert.True(count >= 4);

        // Find top-left
        bool foundTopLeft = false;

        for (int r = 0; r < cpuCorners.Rows; r++)
        {
            for (int c = 0; c < cpuCorners.Cols; c++)
            {
                Vec2f pt = cpuCorners.At<Vec2f>(r, c);

                if (Math.Abs(pt.Item0 - 30) < 3 &&
                    Math.Abs(pt.Item1 - 30) < 3)
                {
                    foundTopLeft = true;
                }
            }
        }

        Assert.True(foundTopLeft, "Could not find the top-left corner.");
    }

    [Fact]
    public void HarrisCorner_ComputeTest()
    {
        VerifyCudaSupport();

        // 1. Arrange: 100x100 black image with a 40x40 white square
        using var cpuSrc = new Mat(100, 100, MatType.CV_8UC1, new Scalar(0));
        Cv2.Rectangle(cpuSrc, new Rect(30, 30, 40, 40), new Scalar(255), -1);

        using var gpuSrc = new GpuMat(); gpuSrc.Upload(cpuSrc);
        using var gpuDst = new GpuMat();

        // 2. Act
        using var harris = OpenCvSharp.Cuda.CornernessCriteria.CreateHarrisCorner(MatType.CV_8UC1, blockSize: 3, ksize: 3, k: 0.04);
        harris.Compute(gpuSrc, gpuDst);

        // 3. Download and Assert
        using var cpuDst = new Mat();
        gpuDst.Download(cpuDst);

        Assert.False(cpuDst.Empty());
        Assert.Equal(MatType.CV_32FC1, cpuDst.Type()); // Response map is always float

        // Check corner response at (30, 30). It should be a large positive value.
        float cornerResponse = cpuDst.At<float>(30, 30);
        Assert.True(cornerResponse > 0);

        // Check response in the middle of the square (flat area). It should be near 0.
        float flatResponse = cpuDst.At<byte>(50, 50);
        Assert.InRange(flatResponse, -0.1f, 0.1f);
    }

    [Fact]
    public void HoughCircles_DetectTest()
    {
        VerifyCudaSupport();

        // 1. Arrange: Create a 200x200 black image and draw a white circle
        // Circle center: (100, 100), Radius: 50
        using var cpuSrc = new Mat(200, 200, MatType.CV_8UC1, new Scalar(0));
        Cv2.Circle(cpuSrc, new Point(100, 100), 50, new Scalar(255), 2);

        using var gpuSrc = new GpuMat(); gpuSrc.Upload(cpuSrc);
        using var gpuCircles = new GpuMat();

        // Create detector
        // dp: 1, minDist: 50, cannyThreshold: 100, votesThreshold: 30, minRadius: 10, maxRadius: 100
        using var detector = OpenCvSharp. Cuda.HoughCirclesDetector.Create(1.0f, 50.0f, 100, 30, 10, 100);

        // 2. Act
        detector.Detect(gpuSrc, gpuCircles);

        // 3. Download and Assert
        using var cpuCircles = new Mat();
        gpuCircles.Download(cpuCircles);

        Assert.False(cpuCircles.Empty());

        // Output for CUDA HoughCircles is a 1-row (or 1-col) matrix of CV_32FC3 (x, y, radius)
        Assert.Equal(MatType.CV_32FC3, cpuCircles.Type());

        // Get the first detected circle
        var circlesIndexer = cpuCircles.GetGenericIndexer<Vec3f>();
        Vec3f circle = circlesIndexer[0];

        // Validate that it found our circle approximately at (100, 100) with radius 50
        Assert.InRange(circle.Item0, 95f, 105f); // X
        Assert.InRange(circle.Item1, 95f, 105f); // Y
        Assert.InRange(circle.Item2, 45f, 55f);  // Radius
    }

    [Fact]
    public void HoughLines_DetectTest()
    {
        VerifyCudaSupport();

        // 1. Arrange: Create a 100x100 black image and draw a white horizontal line
        using var cpuSrc = new Mat(100, 100, MatType.CV_8UC1, new Scalar(0));
        // Line at y=50
        Cv2.Line(cpuSrc, new Point(0, 50), new Point(100, 50), new Scalar(255), 1);

        using var gpuSrc = new GpuMat(); gpuSrc.Upload(cpuSrc);
        using var gpuLines = new GpuMat();

        // Create detector: rho=1, theta=1 degree (PI/180), threshold=50
        float theta = (float)Math.PI / 180.0f;
        using var detector = OpenCvSharp.Cuda.HoughLinesDetector.Create(1.0f, theta, 50);

        // 2. Act
        detector.Detect(gpuSrc, gpuLines);

        // 3. Download and Assert
        using var cpuLines = new Mat();
        gpuLines.Download(cpuLines);

        Assert.False(cpuLines.Empty());

        // Output for CUDA HoughLines is a 1-row (or 1-col) matrix of CV_32FC2 (rho, theta)
        Assert.Equal(MatType.CV_32FC2, cpuLines.Type());

        var linesIndexer = cpuLines.GetGenericIndexer<Vec2f>();
        Vec2f line = linesIndexer[0];

        // For a horizontal line at y=50:
        // rho (distance from origin) should be ~50
        // theta (angle) should be ~PI/2 (1.57 rad)
        Assert.InRange(line.Item0, 48f, 52f);
        Assert.InRange(line.Item1, 1.56f, 1.58f);
    }

    [Fact]
    public void HoughSegments_DetectTest()
    {
        VerifyCudaSupport();

        // 1. Arrange: Create a 100x100 black image and draw a white diagonal segment
        using var cpuSrc = new Mat(100, 100, MatType.CV_8UC1, new Scalar(0));
        Point p1 = new Point(10, 10);
        Point p2 = new Point(80, 80);
        Cv2.Line(cpuSrc, p1, p2, new Scalar(255), 3);

        using var cpuEdges = new Mat();
        Cv2.Canny(cpuSrc, cpuEdges, 50, 150);

        

        using var gpuSrc = new GpuMat(); 
        gpuSrc.Upload(cpuEdges);
        using var gpuLines = new GpuMat();

        // Create detector: rho=1, theta=1 deg, minLength=30, maxGap=10
        float theta = (float)Math.PI / 180.0f;
        using var detector = OpenCvSharp.Cuda.HoughSegmentDetector.Create(
       rho: 1.0f,
       theta: theta,
       minLineLength: 10,   
       maxLineGap: 50       
   );

        // 2. Act
        detector.Detect(gpuSrc, gpuLines);
       
        // 3. Download and Assert
        using var cpuLines = new Mat();
        gpuLines.Download(cpuLines);

        Assert.False(cpuLines.Empty());

        // Output for Segment Detector is a matrix of CV_32SC4 (x1, y1, x2, y2)
        Assert.Equal(MatType.CV_32SC4, cpuLines.Type());

        var linesIndexer = cpuLines.GetGenericIndexer<Vec4i>();
        bool foundDiagonal = false;

        int total = cpuLines.Rows * cpuLines.Cols;
        for (int r = 0; r < cpuLines.Rows; r++)
        {
            for (int c = 0; c < cpuLines.Cols; c++)
            {
                Vec4i s = cpuLines.At<Vec4i>(r, c);

                float dx = s.Item2 - s.Item0;
                float dy = s.Item3 - s.Item1;

                // slope ≈ 1
                if (Math.Abs(Math.Abs(dx) - Math.Abs(dy)) < 5)
                {
                    // length large enough
                    float length = (float)Math.Sqrt(dx * dx + dy * dy);

                    if (length > 50)
                    {
                        foundDiagonal = true;
                        break;
                    }
                }
            }
        }

        Assert.True(foundDiagonal, "No sufficiently long diagonal segment detected.");
    }

    [Fact]
    public void TemplateMatching_MatchTest()
    {
        VerifyCudaSupport();

        using var scene = new Mat(100, 100, MatType.CV_8UC1, new Scalar(0));
        Cv2.Rectangle(scene, new Rect(50, 50, 10, 10), new Scalar(255), -1);

        using var templ = new Mat(10, 10, MatType.CV_8UC1, new Scalar(0));
        Cv2.Rectangle(templ, new Rect(2, 2, 6, 6), new Scalar(255), -1);

        using var gpuScene = new GpuMat(); gpuScene.Upload(scene);
        using var gpuTempl = new GpuMat(); gpuTempl.Upload(templ);
        using var gpuResult = new GpuMat();

        // 2. Act
        using var matcher = OpenCvSharp.Cuda.TemplateMatching.Create(
            MatType.CV_8UC1,
            TemplateMatchModes.CCoeffNormed);
        matcher.Match(gpuScene, gpuTempl, gpuResult);

        // 3. Assert
        using var cpuResult = new Mat();
        gpuResult.Download(cpuResult);

        Cv2.MinMaxLoc(cpuResult, out _, out _, out _, out var maxLoc);

        // allow small tolerance (CUDA can shift peaks slightly)
        Assert.InRange(maxLoc.X, 48, 52);
        Assert.InRange(maxLoc.Y, 48, 52);
    }

    [Fact]
    public void CvtColor_BGR2GRAYTest()
    {
        VerifyCudaSupport();

        // 1. Arrange: Create a 3-channel BGR image (10x10)
        using var cpuSrc = new Mat(10, 10, MatType.CV_8UC3, new Scalar(255, 0, 0)); // Pure Blue
        using var gpuSrc = new GpuMat(); gpuSrc.Upload(cpuSrc);
        using var gpuDst = new GpuMat();

        // 2. Act: Convert BGR to Gray
        Cv2.Cuda.CvtColor(gpuSrc, gpuDst, ColorConversionCodes.BGR2GRAY);

        // 3. Download and Assert
        using var cpuDst = new Mat();
        gpuDst.Download(cpuDst);

        Assert.False(cpuDst.Empty());

        // Source was 3 channels, destination should be 1 channel
        Assert.Equal(3, cpuSrc.Channels());
        Assert.Equal(1, cpuDst.Channels());

        // Check that it's no longer zero
        Assert.NotEqual(0, cpuDst.At<byte>(0, 0));
    }

    [Fact]
    public void Demosaicing_BayerRG2BGRTest()
    {
        VerifyCudaSupport();

        // 1. Arrange: Create a 4x4 Bayer RG pattern
        // A very simple pattern where we set 'Red' and 'Blue' pixels
        using var cpuSrc = new Mat(4, 4, MatType.CV_8UC1, new Scalar(0));
        cpuSrc.Set<byte>(0, 0, 255); // Top-left is Red in BayerRG

        using var gpuSrc = new GpuMat(); gpuSrc.Upload(cpuSrc);
        using var gpuDst = new GpuMat();

        // 2. Act: Convert Bayer RG to BGR (3 channels)
        Cv2.Cuda.Demosaicing(gpuSrc, gpuDst, ColorConversionCodes.BayerRG2BGR);

        // 3. Download and Assert
        using var cpuDst = new Mat();
        gpuDst.Download(cpuDst);

        Assert.False(cpuDst.Empty());

        // Result should have 3 channels (BGR)
        Assert.Equal(3, cpuDst.Channels());

        // In BayerRG2BGR, the pixel at (0,0) in the source affects the 'Red' channel 
        // of the output at (0,0). BGR Vec3b: Item0=B, Item1=G, Item2=R.
        Vec3b pixel = cpuDst.At<Vec3b>(0, 0);
        Assert.True(pixel.Item2 > 0, "Red channel should have been interpolated.");
    }

    [Fact]
    public void EqualizeHist_Test()
    {
        VerifyCudaSupport();

        // 1. Arrange: 100x100 dark gray image (Value = 50)
        using var cpuSrc = new Mat(100, 100, MatType.CV_8UC1);

        for (int y = 0; y < 100; y++)
        {
            for (int x = 0; x < 100; x++)
            {
                cpuSrc.Set<byte>(y, x, (byte)(x));
            }
        }


        using var gpuSrc = new GpuMat(); gpuSrc.Upload(cpuSrc);
        using var gpuDst = new GpuMat();

        // 2. Act
        Cv2.Cuda.EqualizeHist(gpuSrc, gpuDst);

        // 3. Download and Assert
        using var cpuDst = new Mat();
        gpuDst.Download(cpuDst);

        Assert.False(cpuDst.Empty());
        Assert.Equal(cpuSrc.Size(), cpuDst.Size());
        Assert.Equal(MatType.CV_8UC1, cpuDst.Type());

        double minVal, maxVal;
        Cv2.MinMaxLoc(cpuDst, out minVal, out maxVal);

        Assert.True(maxVal - minVal > 0);
    }

    [Fact]
    public void EvenLevels_Test()
    {
        VerifyCudaSupport();

        // 1. Act: Create 5 levels from 0 to 100
        using var gpuLevels = new GpuMat();
        Cv2.Cuda.EvenLevels(gpuLevels,nLevels: 5, lowerLevel: 0, upperLevel: 100);
        // 2. Download and Assert
        using var cpuLevels = new Mat();
        gpuLevels.Download(cpuLevels);

        Assert.False(cpuLevels.Empty());

        // evenLevels outputs a 1D matrix of 32-bit integers (CV_32SC1)
        Assert.Equal(MatType.CV_32SC1, cpuLevels.Type());
        Assert.Equal(5, cpuLevels.Cols * cpuLevels.Rows);

        // Verify the linear distribution
        // (100 - 0) / (5 - 1) = 25 step size
        Assert.Equal(0, cpuLevels.At<int>(0, 0));
        Assert.Equal(25, cpuLevels.At<int>(0, 1));
        Assert.Equal(50, cpuLevels.At<int>(0, 2));
        Assert.Equal(75, cpuLevels.At<int>(0, 3));
        Assert.Equal(100, cpuLevels.At<int>(0, 4));
    }

    [Fact]
    public void FastNlMeansDenoising_Test()
    {
        VerifyCudaSupport();

        // 1. Arrange: 50x50 gray image (Value 100) with a noise pixel (Value 200) at center
        using var cpuSrc = new Mat(50, 50, MatType.CV_8UC1, new Scalar(100));
        cpuSrc.Set<byte>(25, 25, 200);

        using var gpuSrc = new GpuMat(); gpuSrc.Upload(cpuSrc);
        using var gpuDst = new GpuMat();

        // 2. Act: Apply NL Means with a strong filter strength (h=50)
        Cv2.Cuda.FastNlMeansDenoising(gpuSrc, gpuDst, h: 50.0f);

        // 3. Download and Assert
        using var cpuDst = new Mat();
        gpuDst.Download(cpuDst);

        Assert.False(cpuDst.Empty());
        Assert.Equal(MatType.CV_8UC1, cpuDst.Type());

        byte originalNoise = 200;
        byte denoisedPixel = cpuDst.At<byte>(25, 25);

        // The denoising process should have pulled the value 200 much closer to 100
        Assert.True(denoisedPixel < originalNoise, $"Expected noise to be reduced, but was {denoisedPixel}");
    }
}



