using System;
using System.Collections.Generic;
using System.Security.Cryptography;
using System.Text;
using OpenCvSharp.Cuda;
using OpenCvSharp.Internal;
using OpenCvSharp.Modules.core.Enum;

namespace OpenCvSharp;

public static partial class Cv2
{
    public static partial class Cuda
    {
        /// <summary>
        /// Colors a disparity image.
        /// </summary>
        /// <param name="srcDisp">Source disparity image (1-channel).</param>
        /// <param name="dstDisp">Destination colored disparity image (4-channel, CV_8UC4).</param>
        /// <param name="ndisp">Number of disparities.</param>
        /// <param name="stream">Stream for the asynchronous version.</param>
        public static void DrawColorDisp(OpenCvSharp.Cuda.InputArray srcDisp, OpenCvSharp.Cuda.OutputArray dstDisp, int ndisp, OpenCvSharp.Cuda.Stream? stream = null)
        {
            if (srcDisp is null) 
                throw new ArgumentNullException(nameof(srcDisp));
            if (dstDisp is null) 
                throw new ArgumentNullException(nameof(dstDisp));

            srcDisp.ThrowIfDisposed();
            dstDisp.ThrowIfNotReady();

            NativeMethods.HandleException(
                NativeMethods.cuda_drawColorDisp(srcDisp.CvPtr, dstDisp.CvPtr, ndisp, ToPtr(stream)));

            GC.KeepAlive(srcDisp);
            dstDisp.Fix();
         
        }
    }
}
