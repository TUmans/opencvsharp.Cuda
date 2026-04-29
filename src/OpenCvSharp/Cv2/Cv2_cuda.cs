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
