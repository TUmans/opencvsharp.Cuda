#if ENABLED_CUDA

using System;
using System.Collections.Generic;
using OpenCvSharp.Cuda;
using OpenCvSharp.Internal;

namespace OpenCvSharp
{
    public static partial class Cv2
    {
        public static partial class Cuda
        {
            #region Hardware

           

        

            /// <summary>
            /// Sets a device and initializes it for the current thread.
            /// </summary>
            /// <param name="device">System index of a GPU device starting with 0.</param>
            /// <returns></returns>
            public static int SetDevice(int device)
            {
                ThrowIfGpuNotAvailable();
                NativeMethods.HandleException(NativeMethods.cuda_getDevice(out int res));
                return res;
            }

            /// <summary>
            /// Explicitly destroys and cleans up all resources associated with the current device in the current process.
            /// Any subsequent API call to this device will reinitialize the device.
            /// </summary>
            public static void ResetDevice()
            {
                ThrowIfGpuNotAvailable();
                NativeMethods.cuda_resetDevice();
            }

            /// <summary>
            /// 
            /// </summary>
            /// <param name="device"></param>
            public static void PrintCudaDeviceInfo(int device)
            {
                ThrowIfGpuNotAvailable();
                NativeMethods.cuda_printCudaDeviceInfo(device);
            }

            /// <summary>
            /// 
            /// </summary>
            /// <param name="device"></param>
            public static void PrintShortCudaDeviceInfo(int device)
            {
                ThrowIfGpuNotAvailable();
                NativeMethods.cuda_printShortCudaDeviceInfo(device);
            }

            /// <summary>
            /// Checks whether the current device supports the given feature.
            /// </summary>
            /// <param name="featureSet">Feature set to check.</param>
            /// <returns>True if supported.</returns>
            public static bool DeviceSupports(FeatureSet featureSet)
            {
                ThrowIfGpuNotAvailable();

                NativeMethods.HandleException(
                    NativeMethods.cuda_deviceSupports((int)featureSet, out var ret));

                return ret != 0;
            }

            #endregion

            #region CudaMem

            /// <summary>
            /// Page-locks the matrix m memory and maps it for the device(s)
            /// </summary>
            /// <param name="m"></param>
            public static void RegisterPageLocked(Mat m)
            {
                ThrowIfGpuNotAvailable();
                if (m is null)
                    throw new ArgumentNullException(nameof(m));
                NativeMethods.cuda_registerPageLocked(m.CvPtr);
                GC.KeepAlive(m);
            }

            /// <summary>
            /// Unmaps the memory of matrix m, and makes it pageable again.
            /// </summary>
            /// <param name="m"></param>
            public static void UnregisterPageLocked(Mat m)
            {
                ThrowIfGpuNotAvailable();
                if (m is null)
                    throw new ArgumentNullException(nameof(m));
                NativeMethods.cuda_unregisterPageLocked(m.CvPtr);
                GC.KeepAlive(m);
            }

            #endregion

            #region GpuMat

            /// <summary>
            /// Creates continuous GPU matrix
            /// </summary>
            /// <param name="rows">Number of rows in a 2D array.</param>
            /// <param name="cols">Number of columns in a 2D array.</param>
            /// <param name="type">Array type.</param>
            /// <param name="m"></param>
            public static void CreateContinuous(int rows, int cols, MatType type, OpenCvSharp.Cuda.OutputArray m)
            {
                ThrowIfGpuNotAvailable();
                if (m is null)
                    throw new ArgumentNullException(nameof(m));
                NativeMethods.cuda_createContinuous1(rows, cols, (int)type, m.CvPtr);
                GC.KeepAlive(m);
            }

            /// <summary>
            /// Creates continuous GPU matrix
            /// </summary>
            /// <param name="rows">Number of rows in a 2D array.</param>
            /// <param name="cols">Number of columns in a 2D array.</param>
            /// <param name="type">Array type.</param>
            /// <returns></returns>
            public static OpenCvSharp.Cuda.OutputArray CreateContinuous(int rows, int cols, MatType type)
            {
                ThrowIfGpuNotAvailable();
                NativeMethods.HandleException(NativeMethods.cuda_createContinuous2(rows, cols, (int)type, out IntPtr ret));
                return new GpuMat(ret);
            }

            /// <summary>
            /// Creates continuous GPU matrix
            /// </summary>
            /// <param name="size">Number of rows and columns in a 2D array.</param>
            /// <param name="type">Array type.</param>
            /// <param name="m"></param>
            public static void CreateContinuous(Size size, MatType type, OpenCvSharp.Cuda.OutputArray m)
            {
                ThrowIfGpuNotAvailable();
                CreateContinuous(size.Height, size.Width, type, m);
            }

            /// <summary>
            /// Creates continuous GPU matrix
            /// </summary>
            /// <param name="size">Number of rows and columns in a 2D array.</param>
            /// <param name="type">Array type.</param>
            /// <returns></returns>
            public static OpenCvSharp.Cuda.OutputArray CreateContinuous(Size size, MatType type)
            {
                ThrowIfGpuNotAvailable();
                return CreateContinuous(size.Height, size.Width, type);
            }

         

            #endregion

           
        }
    }
}




#endif
