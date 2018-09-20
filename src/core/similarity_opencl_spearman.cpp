#include "similarity_opencl_spearman.h"



using namespace std;






/*!
 * Construct a new Spearman kernel object with the given OpenCL program and
 * qt parent.
 *
 * @param program
 * @param parent
 */
Similarity::OpenCL::Spearman::Spearman(::OpenCL::Program* program, QObject* parent):
   ::OpenCL::Kernel(program, "Spearman_compute", parent)
{
   EDEBUG_FUNC(this,parent);
}






/*!
 * Execute this kernel object's OpenCL kernel using the given OpenCL command
 * queue and kernel arguments, returning the OpenCL event associated with the
 * kernel execution.
 *
 * @param queue
 * @param kernelSize
 * @param in_data
 * @param clusterSize
 * @param in_labels
 * @param sampleSize
 * @param minSamples
 * @param work_x
 * @param work_y
 * @param work_rank
 * @param out_correlations
 */
::OpenCL::Event Similarity::OpenCL::Spearman::execute(
   ::OpenCL::CommandQueue* queue,
   int kernelSize,
   ::OpenCL::Buffer<Pairwise::Vector2>* in_data,
   cl_char clusterSize,
   ::OpenCL::Buffer<cl_char>* in_labels,
   cl_int sampleSize,
   cl_int minSamples,
   ::OpenCL::Buffer<cl_float>* work_x,
   ::OpenCL::Buffer<cl_float>* work_y,
   ::OpenCL::Buffer<cl_int>* work_rank,
   ::OpenCL::Buffer<cl_float>* out_correlations
)
{
   EDEBUG_FUNC(this,
      queue,
      kernelSize,
      in_data,
      clusterSize,
      in_labels,
      sampleSize,
      minSamples,
      work_x,
      work_y,
      work_rank,
      out_correlations);

   // acquire lock for this kernel
   Locker locker {lock()};

   // set kernel arguments
   setBuffer(InData, in_data);
   setArgument(ClusterSize, clusterSize);
   setBuffer(InLabels, in_labels);
   setArgument(SampleSize, sampleSize);
   setArgument(MinSamples, minSamples);
   setBuffer(WorkX, work_x);
   setBuffer(WorkY, work_y);
   setBuffer(WorkRank, work_rank);
   setBuffer(OutCorrelations, out_correlations);

   // set kernel sizes
   setSizes(0, kernelSize, min(kernelSize, maxWorkGroupSize(queue->device())));

   // execute kernel
   return ::OpenCL::Kernel::execute(queue);
}
