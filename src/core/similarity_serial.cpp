#include "similarity_serial.h"
#include "similarity_resultblock.h"
#include "similarity_workblock.h"
#include "expressionmatrix_gene.h"



using namespace std;






Similarity::Serial::Serial(Similarity* parent):
   EAbstractAnalytic::Serial(parent),
   _base(parent)
{
   // initialize clustering model
   if ( _base->_clusMethod != ClusteringMethod::None )
   {
      _base->_clusModel->initialize(_base->_input);
   }

   // initialize correlation model
   _base->_corrModel->initialize(_base->_input);
}






std::unique_ptr<EAbstractAnalytic::Block> Similarity::Serial::execute(const EAbstractAnalytic::Block* block)
{
   // cast block to work block
   const WorkBlock* workBlock {block->cast<WorkBlock>()};

   // initialize result block
   ResultBlock* resultBlock {new ResultBlock(workBlock->index(), workBlock->start())};

   // initialize workspace
   QVector<Pairwise::Vector2> X(_base->_input->sampleSize());
   QVector<qint8> labels(_base->_input->sampleSize());

   // iterate through all pairs
   Pairwise::Index index {workBlock->start()};

   for ( int i = 0; i < workBlock->size(); ++i )
   {
      // fetch pairwise input data
      int numSamples = fetchPair(index, X, labels);

      // remove pre-clustering outliers
      if ( _base->_removePreOutliers )
      {
         markOutliers(X, numSamples, 0, labels, 0, -7);
         markOutliers(X, numSamples, 1, labels, 0, -7);
      }

      // compute clusters
      qint8 K {1};

      if ( _base->_clusMethod != ClusteringMethod::None )
      {
         K = _base->_clusModel->compute(
            X,
            numSamples,
            labels,
            _base->_minSamples,
            _base->_minClusters,
            _base->_maxClusters,
            _base->_criterion
         );
      }

      // remove post-clustering outliers
      if ( K > 1 && _base->_removePostOutliers )
      {
         for ( qint8 k = 0; k < K; ++k )
         {
            markOutliers(X, numSamples, 0, labels, k, -8);
            markOutliers(X, numSamples, 1, labels, k, -8);
         }
      }

      // compute correlations
      QVector<float> correlations = _base->_corrModel->compute(
         X,
         K,
         labels,
         _base->_minSamples
      );

      // save pairwise output data
      Pair pair;
      pair.K = K;

      if ( K > 1 )
      {
         pair.labels = labels;
      }

      if ( K > 0 )
      {
         pair.correlations = correlations;
      }

      resultBlock->append(pair);

      // increment to next pair
      ++index;
   }

   // return result block
   return unique_ptr<EAbstractAnalytic::Block>(resultBlock);
}






int Similarity::Serial::fetchPair(Pairwise::Index index, QVector<Pairwise::Vector2>& X, QVector<qint8>& labels)
{
   // read in gene expressions
   ExpressionMatrix::Gene gene1(_base->_input);
   ExpressionMatrix::Gene gene2(_base->_input);

   gene1.read(index.getX());
   gene2.read(index.getY());

   // populate X with shared expressions of gene pair
   int numSamples = 0;

   for ( int i = 0; i < _base->_input->sampleSize(); ++i )
   {
      if ( std::isnan(gene1.at(i)) || std::isnan(gene2.at(i)) )
      {
         labels[i] = -9;
      }
      else if ( gene1.at(i) < _base->_minExpression || gene2.at(i) < _base->_minExpression )
      {
         labels[i] = -6;
      }
      else
      {
         X[numSamples] = { gene1.at(i), gene2.at(i) };
         numSamples++;

         labels[i] = 0;
      }
   }

   // return size of X
   return numSamples;
}






void Similarity::Serial::markOutliers(const QVector<Pairwise::Vector2>& X, int N, int j, QVector<qint8>& labels, qint8 cluster, qint8 marker)
{
   // compute x_sorted = X[:, j], filtered and sorted
   QVector<float> x_sorted;
   x_sorted.reserve(N);

   for ( int i = 0; i < N; i++ )
   {
      if ( labels[i] == cluster || labels[i] == marker )
      {
         x_sorted.append(X[i].s[j]);
      }
   }

   if ( x_sorted.size() == 0 )
   {
      return;
   }

   std::sort(x_sorted.begin(), x_sorted.end());

   // compute quartiles, interquartile range, upper and lower bounds
   const int n = x_sorted.size();

   float Q1 = x_sorted[n * 1 / 4];
   float Q3 = x_sorted[n * 3 / 4];

   float T_min = Q1 - 1.5f * (Q3 - Q1);
   float T_max = Q3 + 1.5f * (Q3 - Q1);

   // mark outliers
   for ( int i = 0; i < N; ++i )
   {
      if ( labels[i] == cluster && (X[i].s[j] < T_min || T_max < X[i].s[j]) )
      {
         labels[i] = marker;
      }
   }
}
