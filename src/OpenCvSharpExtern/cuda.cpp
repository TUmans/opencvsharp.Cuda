// ReSharper disable CppUnusedIncludeDirective

#ifdef ENABLED_CUDA
#include "cuda.h"
#include "cuda_Core.h"
#include "cuda_GpuMat.h"
#include "cuda_Arithm.h"
#include "cuda_Imgproc.h"
#include "cuda_Warping.h"
#include "cuda_Legacy.h"
#include "cuda_Filters.h"
#include "cuda_Stereo.h"
#include "cuda_OpenGL.h"
#include "cuda_Photo.h"
#include "cuda_Stream.h"
#include "cuda_Event.h"

#include "cuda_Bgsegm_mog.h"
#include "cuda_Bgsegm_mog2.h"

#include "cuda_BufferPool.h"

// arithm
#include "cuda_Convolution.h"
#include "cuda_DFT.h"
#include "cuda_LookUpTable.h"

// imgproc
#include "cuda_CannyEdgeDetector.h"
#include "cuda_CLAHE.h"
#include "cuda_CascadeClassifier.h"
#include "cuda_CornernessCriteria.h"
#include "cuda_CornersDetector.h"
#include "cuda_GeneralizedHough.h"
#include "cuda_GeneralizedHoughBallard.h"
#include "cuda_GeneralizedHoughGuil.h"
#include "cuda_HoughCirclesDetector.h"
#include "cuda_HoughLinesDetector.h"
#include "cuda_HoughSegmentDetector.h"
#include "cuda_TemplateMatching.h"

// legacy
#include "cuda_Bgsegm_gmg.h"
#include "cuda_Bgsegm_fgd.h"
#include "cuda_ImagePyramid.h"

//objectdetect
#include "cuda_CascadeClassifier.h"

//optiflow
#include "cuda_BroxOpticalFlow.h"
#include "cuda_DenseOpticalFlow.h"

// stereo
#include "cuda_DisparityBilateralFilter.h"
#include "cuda_StereoBeliefPropagation.h"
#include "cuda_StereoBM.h"
#include "cuda_StereoConstantSpaceBP.h"
#include "cuda_StereoMatcher.h"
#include "cuda_StereoSGM.h"
#endif
