#if ENABLED_CUDA

using System;
using System.Collections.Generic;
using System.Text;
using OpenCvSharp.Internal;

namespace OpenCvSharp.Cuda
{
    /// <summary>
    /// Gives information about the given GPU
    /// </summary>
    public sealed class DeviceInfo : DisposableGpuObject
    {
        /// <summary>
        /// Creates DeviceInfo object for the current GPU
        /// </summary>
        public DeviceInfo()
        {
            Cv2.ThrowIfGpuNotAvailable();
            NativeMethods.HandleException(NativeMethods.cuda_DeviceInfo_new1(out IntPtr p));
            InitSafeHandle(p);
        }

        /// <summary>
        /// Creates DeviceInfo object for the given GPU
        /// </summary>
        /// <param name="deviceId"></param>
        public DeviceInfo(int deviceId)
        {
            Cv2.ThrowIfGpuNotAvailable();
            NativeMethods.HandleException(NativeMethods.cuda_DeviceInfo_new2(deviceId, out IntPtr p));
            InitSafeHandle(p);
        }

        /// <summary>
        /// Releases unmanaged resources
        /// </summary>

        private void InitSafeHandle(IntPtr p, bool ownsHandle = true)
        {
            SetSafeHandle(new OpenCvPtrSafeHandle(p, ownsHandle,
                static h => NativeMethods.cuda_DeviceInfo_delete(h)));
        }

        /// <summary>
        /// 
        /// </summary>
        public int DeviceId
        {
            get
            {
                NativeMethods.HandleException(NativeMethods.cuda_DeviceInfo_deviceID(CvPtr, out int res));
                GC.KeepAlive(this);
                return res;
            }
        }

        /// <summary>
        /// 
        /// </summary>
        public string Name
        {
            get
            {
                var buf = new StringBuilder(1 << 16);
                NativeMethods.HandleException(NativeMethods.cuda_DeviceInfo_name(CvPtr, buf, buf.Capacity));
                GC.KeepAlive(this);
                return buf.ToString();
            }
        }

        /// <summary>
        /// Return compute capability versions
        /// </summary>
        public int MajorVersion
        {
            get
            {
                NativeMethods.HandleException(NativeMethods.cuda_DeviceInfo_majorVersion(CvPtr, out int res));
                GC.KeepAlive(this);
                return res;
            }
        }

        /// <summary>
        /// Return compute capability versions
        /// </summary>
        public int MinorVersion
        {
            get
            {
                NativeMethods.HandleException(NativeMethods.cuda_DeviceInfo_minorVersion(CvPtr, out int res));
                GC.KeepAlive(this);
                return res;
            }
        }

        /// <summary>
        /// 
        /// </summary>
        public int MultiProcessorCount
        {
            get
            {
                NativeMethods.HandleException(NativeMethods.cuda_DeviceInfo_multiProcessorCount(CvPtr, out int res));
                GC.KeepAlive(this);
                return res;
            }
        }

        /// <summary>
        /// 
        /// </summary>
        public long SharedMemPerBlock
        {
            get
            {
                NativeMethods.HandleException(NativeMethods.cuda_DeviceInfo_sharedMemPerBlock(CvPtr, out ulong res));
                GC.KeepAlive(this);
                return (long)res;
            }
        }

        /// <summary>
        /// 
        /// </summary>
        public void QueryMemory(out long totalMemory, out long freeMemory)
        {
            ulong t, f;
            NativeMethods.HandleException(NativeMethods.cuda_DeviceInfo_queryMemory(CvPtr, out t, out f));
            GC.KeepAlive(this);
            totalMemory = (long)t;
            freeMemory = (long)f;
        }

        /// <summary>
        /// 
        /// </summary>
        public long FreeMemory
        {
            get
            {
                NativeMethods.HandleException(NativeMethods.cuda_DeviceInfo_freeMemory(CvPtr, out ulong res));
                GC.KeepAlive(this);
                return (long)res;
            }
        }

        /// <summary>
        /// 
        /// </summary>
        public long TotalMemory
        {
            get
            {
                NativeMethods.HandleException(NativeMethods.cuda_DeviceInfo_totalMemory(CvPtr, out ulong res));
                GC.KeepAlive(this);
                return (long)res;
            }
        }

        /// <summary>
        /// Checks whether device supports the given feature
        /// </summary>
        /// <param name="featureSet"></param>
        /// <returns></returns>
        public bool Supports(FeatureSet featureSet)
        {
            NativeMethods.HandleException(NativeMethods.cuda_DeviceInfo_supports(CvPtr, (int)featureSet, out int res));
            GC.KeepAlive(this);
            return res != 0;
        }

        /// <summary>
        /// Checks whether the GPU module can be run on the given device
        /// </summary>
        /// <returns></returns>
        public bool IsCompatible
        {
            get
            {
                NativeMethods.HandleException(NativeMethods.cuda_DeviceInfo_isCompatible(CvPtr, out int res));
                GC.KeepAlive(this);
                return res !=0;
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public bool CanMapHostMemory
        {
            get
            {
                NativeMethods.HandleException(NativeMethods.cuda_DeviceInfo_canMapHostMemory(CvPtr, out int res));
                GC.KeepAlive(this);
                return res != 0;
            }
        }
    }
}

#endif
