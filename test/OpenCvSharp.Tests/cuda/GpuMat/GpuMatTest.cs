using System;
using System.Collections.Generic;
using System.Text;
using Xunit;
using OpenCvSharp;

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

   
    private static bool HasCuda()
    {
        try
        {
            return Cv2.GetCudaEnabledDeviceCount() > 0;
        }
        catch
        {
            return false;
        }
    }

    [Fact]
    public void EmptyConstructor()
    {
        if (!HasCuda()) return;

        using var gpuMat = new Cuda.GpuMat();
        Assert.True(gpuMat.Empty());
        Assert.Equal(0, gpuMat.Rows);
        Assert.Equal(0, gpuMat.Cols);
    }

    [Fact]
    public void SizeAndTypeConstructor()
    {
        if (!HasCuda()) return;

        int rows = 10;
        int cols = 20;
        using var gpuMat = new Cuda.GpuMat(rows, cols, MatType.CV_8UC3);

        Assert.False(gpuMat.Empty());
        Assert.Equal(rows, gpuMat.Rows);
        Assert.Equal(cols, gpuMat.Cols);
        Assert.Equal(MatType.CV_8UC3, gpuMat.Type());
        Assert.Equal(3, gpuMat.Channels());
        //Assert.True(gpuMat.IsContinuous());
    }

    [Fact]
    public void ScalarConstructor()
    {
        if (!HasCuda()) return;

        using var gpuMat = new Cuda.GpuMat(5, 5, MatType.CV_8UC1, new Scalar(42));
        using var hostMat = new Mat();

        // Download to host to safely verify values
        gpuMat.Download(hostMat);

        Assert.Equal(42, hostMat.At<byte>(0, 0));
        Assert.Equal(42, hostMat.At<byte>(4, 4));
    }

    [Fact]
    public void UploadDownload()
    {
        if (!HasCuda()) return;

        using var hostMat = new Mat(15, 15, MatType.CV_32FC1, new Scalar(3.14f));
        using var gpuMat = new Cuda.GpuMat();

        // Upload
        gpuMat.Upload(hostMat);
        Assert.False(gpuMat.Empty());
        Assert.Equal(15, gpuMat.Rows);
        Assert.Equal(15, gpuMat.Cols);
        Assert.Equal(MatType.CV_32FC1, gpuMat.Type());

        // Download
        using var hostMat2 = new Mat();
        gpuMat.Download(hostMat2);

        Assert.Equal(3.14f, hostMat2.At<float>(7, 7), 5);
    }

    [Fact]
    public void MatToGpuMatCast()
    {
        if (!HasCuda()) return;

        using var hostMat = new Mat(10, 10, MatType.CV_8UC1, new Scalar(100));
        using var gpuMat = new Cuda.GpuMat(hostMat);

        Assert.Equal(10, gpuMat.Rows);
        Assert.Equal(MatType.CV_8UC1, gpuMat.Type());

        using var hostMat2 = new Mat();
        gpuMat.Download(hostMat2);
        Assert.Equal(100, hostMat2.At<byte>(5, 5));
    }

    [Fact]
    public void Clone()
    {
        if (!HasCuda()) return;

        using var gpuMat1 = new Cuda.GpuMat(10, 10, MatType.CV_8UC1, new Scalar(50));
        using var gpuMat2 = gpuMat1.Clone();

        Assert.Equal(gpuMat1.Rows, gpuMat2.Rows);
        Assert.Equal(gpuMat1.Cols, gpuMat2.Cols);
        Assert.NotEqual(gpuMat1.Data, gpuMat2.Data); // Should be a deep copy

        using var hostMat = new Mat();
        gpuMat2.Download(hostMat);
        Assert.Equal(50, hostMat.At<byte>(0, 0));
    }

    [Fact]
    public void CopyTo()
    {
        if (!HasCuda()) return;

        using var src = new Cuda.GpuMat(5, 5, MatType.CV_8UC1, new Scalar(100));
        using var dst = new Cuda.GpuMat();

        src.CopyTo(dst);

        Assert.Equal(src.Size(), dst.Size());
        Assert.Equal(src.Type(), dst.Type());

        using var hostDst = new Mat();
        dst.Download(hostDst);
        Assert.Equal(100, hostDst.At<byte>(2, 2));
    }

    [Fact]
    public void ConvertTo()
    {
        if (!HasCuda()) return;

        using var src = new Cuda.GpuMat(5, 5, MatType.CV_8UC1, new Scalar(10));
        using var dst = new Cuda.GpuMat();

        // Convert byte to float, and scale by 2.5
        src.ConvertTo(dst, MatType.CV_32FC1, 2.5, 0);

        Assert.Equal(MatType.CV_32FC1, dst.Type());

        using var hostDst = new Mat();
        dst.Download(hostDst);
        Assert.Equal(25.0f, hostDst.At<float>(0, 0), 5);
    }

    [Fact]
    public void SetTo()
    {
        if (!HasCuda()) return;

        using var gpuMat = new Cuda.GpuMat(5, 5, MatType.CV_8UC1, new Scalar(0));
        gpuMat.SetTo(new Scalar(255));

        using var hostMat = new Mat();
        gpuMat.Download(hostMat);
        Assert.Equal(255, hostMat.At<byte>(3, 3));
    }

    //[Fact]
    // https://github.com/opencv/opencv/issues/4728
    //public void Reshape()
    //{
    //    if (!HasCuda()) return;

    //    // 10x10, 1 channel
    //    using var gpuMat = new Cuda.GpuMat(10, 10, MatType.CV_8UC1);

    //    // Reshape to 2 channels, calculate rows automatically (0 means keep same if possible, or calculate based on channels)
    //    // 100 elements / 2 channels = 50 elements. If we say rows = 5, cols will be 10.
    //    using var reshaped = gpuMat.Reshape(2, 5);

    //    Assert.Equal(5, reshaped.Rows);
    //    Assert.Equal(10, reshaped.Cols);
    //    Assert.Equal(2, reshaped.Channels());
    //}

    [Fact]
    public void SubMatrixROI()
    {
        if (!HasCuda()) return;

        using var hostMat = new Mat(10, 10, MatType.CV_8UC1, new Scalar(0));
        // Set a 2x2 block in the middle to 255
        hostMat.SubMat(new Rect(4, 4, 2, 2)).SetTo(new Scalar(255));

        using var gpuMat = new Cuda.GpuMat();
        gpuMat.Upload(hostMat);

        // Extract ROI on GPU
        using var roiGpu = new Cuda.GpuMat(gpuMat, new Rect(4, 4, 2, 2));

        Assert.Equal(2, roiGpu.Rows);
        Assert.Equal(2, roiGpu.Cols);

        using var roiHost = new Mat();
        roiGpu.Download(roiHost);

        Assert.Equal(255, roiHost.At<byte>(0, 0));
        Assert.Equal(255, roiHost.At<byte>(1, 1));
    }

    [Fact]
    public void RowAndColRange()
    {
        if (!HasCuda()) return;

        using var gpuMat = new Cuda.GpuMat(10, 10, MatType.CV_8UC1);

        using var rowRange = gpuMat.RowRange(2, 5);
        Assert.Equal(3, rowRange.Rows);
        Assert.Equal(10, rowRange.Cols);

        using var colRange = gpuMat.ColRange(new Range(3, 7));
        Assert.Equal(10, colRange.Rows);
        Assert.Equal(4, colRange.Cols);
    }

    [Fact]
    public void Swap()
    {
        if (!HasCuda()) return;

        using var m1 = new Cuda.GpuMat(5, 5, MatType.CV_8UC1);
        using var m2 = new Cuda.GpuMat(10, 10, MatType.CV_32FC1);

        m1.Swap(m2);

        Assert.Equal(10, m1.Rows);
        Assert.Equal(MatType.CV_32FC1, m1.Type());

        Assert.Equal(5, m2.Rows);
        Assert.Equal(MatType.CV_8UC1, m2.Type());
    }

    [Fact]
    public void PropertiesTest()
    {
        if (!HasCuda()) return;

        using var m = new Cuda.GpuMat(10, 20, MatType.CV_32FC3);

        Assert.Equal(10, m.Rows);
        Assert.Equal(20, m.Cols);
        Assert.Equal(10, m.Height);
        Assert.Equal(20, m.Width);
        Assert.Equal(3, m.Channels());
        Assert.Equal(MatType.CV_32FC3, m.Type());

        Assert.True(m.Step() > 0);
        Assert.True(m.ElemSize() > 0);
        Assert.True(m.ElemSize1() > 0);

        Assert.NotEqual(IntPtr.Zero, m.Data);
        Assert.NotEqual(IntPtr.Zero, m.DataStart);
        Assert.NotEqual(IntPtr.Zero, m.DataEnd);
    }

    [Fact]
    public void LocateROI()
    {
        if (!HasCuda()) return;

        using var gpuMat = new Cuda.GpuMat(20, 20, MatType.CV_8UC1);
        using var roi = new Cuda.GpuMat(gpuMat, new Rect(5, 5, 10, 10));

        roi.LocateROI(out Size wholeSize, out Point ofs);

        Assert.Equal(20, wholeSize.Width);
        Assert.Equal(20, wholeSize.Height);
        Assert.Equal(5, ofs.X);
        Assert.Equal(5, ofs.Y);
    }

    
}
