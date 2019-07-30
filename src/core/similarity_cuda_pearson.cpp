#include "similarity_cuda_pearson.h"



using namespace std;






/*!
 * Construct a new Pearson kernel object with the given CUDA program.
 *
 * @param program
 */
Similarity::CUDA::Pearson::Pearson(::CUDA::Program* program):
   ::CUDA::Kernel(program, "Pearson_compute")
{
   EDEBUG_FUNC(this,program);
}






/*!
 * Execute this kernel object's CUDA kernel using the given CUDA stream
 * and kernel arguments, returning the CUDA event associated with the
 * kernel execution.
 *
 * @param stream
 * @param globalWorkSize
 * @param localWorkSize
 * @param numPairs
 * @param expressions
 * @param sampleSize
 * @param in_index
 * @param in_argsort
 * @param clusterSize
 * @param in_labels
 * @param minSamples
 * @param out_correlations
 */
::CUDA::Event Similarity::CUDA::Pearson::execute(
   const ::CUDA::Stream& stream,
   int globalWorkSize,
   int localWorkSize,
   int numPairs,
   ::CUDA::Buffer<float>* expressions,
   int sampleSize,
   ::CUDA::Buffer<int2>* in_index,
   ::CUDA::Buffer<int>* in_argsort,
   char clusterSize,
   ::CUDA::Buffer<qint8>* in_labels,
   int minSamples,
   ::CUDA::Buffer<float>* out_correlations
)
{
   EDEBUG_FUNC(this,
      &stream,
      globalWorkSize,
      localWorkSize,
      numPairs,
      expressions,
      sampleSize,
      in_index,
      in_argsort,
      clusterSize,
      in_labels,
      minSamples,
      out_correlations);

   // set kernel arguments
   setArgument(NumPairs, numPairs);
   setBuffer(Expressions, expressions);
   setArgument(SampleSize, sampleSize);
   setBuffer(InIndex, in_index);
   setBuffer(InArgsort, in_argsort);
   setArgument(ClusterSize, clusterSize);
   setBuffer(InLabels, in_labels);
   setArgument(MinSamples, minSamples);
   setBuffer(OutCorrelations, out_correlations);

   // set work sizes
   setSizes(globalWorkSize / localWorkSize, localWorkSize);

   // execute kernel
   return ::CUDA::Kernel::execute(stream);
}
