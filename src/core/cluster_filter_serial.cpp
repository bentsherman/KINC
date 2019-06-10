#include "cluster_filter_serial.h"
#include "cluster_filter_resultblock.h"
#include "cluster_filter_workblock.h"
#include "expressionmatrix_gene.h"
#include "pairwise_gmm.h"
#include "pairwise_pearson.h"
#include "pairwise_spearman.h"
#include <ace/core/elog.h>
#include "stats.hpp"


using namespace std;


/*!
 * Construct a new serial object with the given analytic as its parent.
 *
 * @param parent
 */
ClusterFilter::Serial::Serial(ClusterFilter* parent):
   EAbstractAnalyticSerial(parent),
   _base(parent)
{
   EDEBUG_FUNC(this,parent);

}


/*!
 * Read in the given work block and save the results in a new result block. This
 * implementation takes the starting pairwise index and pair size from the work
 * block and processes those pairs.
 *
 * @param block
 */
std::unique_ptr<EAbstractAnalyticBlock> ClusterFilter::Serial::execute(const EAbstractAnalyticBlock* block)
{
   EDEBUG_FUNC(this,block);

   if ( ELog::isActive() )
   {
      ELog() << tr("Executing(serial) work index %1.\n").arg(block->index());
   }

   // Cast block to work block.
   const WorkBlock* workBlock {block->cast<WorkBlock>()};

   // Initialize result block.
   ResultBlock* resultBlock {new ResultBlock(workBlock->index(), workBlock->start())};

   // Create iterators for the CCM and CMX data objects.
   CCMatrix::Pair ccmPair = CCMatrix::Pair(_base->_ccm);
   CorrelationMatrix::Pair cmxPair = CorrelationMatrix::Pair(_base->_cmx);

   // Move to the location in the CCM/CMX matrix where the work block starts.
   Pairwise::Index index {workBlock->start()};
   cmxPair.read(index);
   ccmPair.read(cmxPair.index());

   // Iterate through the elements in the workblock.
   qint64 block_size = workBlock->size();
   for ( qint64 i = 0; i < block_size; ++i )
   {
       // Get the number of samples and clusters.
       int num_clusters = ccmPair.clusterSize();
       int num_samples = _base->_emx->sampleSize();

       // This will store how many clusters will remain.
       qint8 num_final_K = 0;

       // Initialize new correlation and labels lists
       QVector<float> new_correlations;
       QVector<qint8> new_labels;
       QVector<int> k_num_samples(num_clusters, 0);

       // Get the list of correlations
       QVector<float> correlations = cmxPair.correlations();

       // Rebuild the pair labels from the pair sample mask.
       QVector<qint8> labels(num_samples, 0);
       for ( qint8 k = 0; k < num_clusters; k++ ) {
          for ( int j = 0; j < num_samples; j++ ) {
             qint8 val = ccmPair.at(k, j);
             if (val == 1) {
               labels[j] = k;
               k_num_samples[k]++;
             }
             else {
               labels[j] = -val;
             }
          }
       }

       // Iterate through the clusters and perform a correlation power analysis.
       for ( qint8 k = 0; k < num_clusters; k++ ) {

           // Do the correlation power test. The following code is modeled
           // after the `pwr.r.test` function of the `pwr` package for R. Here
           // we calculate the power given the signficance level (alpha) provided
           // by the user, the number of samples in the cluster and we
           // compare the calculated power to that expected by the user.
           // This code uses functions from the Keith OHare StatsLib at
           // https://www.kthohr.com/statslib.html
           int n = k_num_samples[k];
           double r = correlations[k];
           double sig_level = _base->_powerThresholdAlpha;
           double ttt = stats::qt(sig_level / 2, n - 2);
           double ttt_2 = pow(ttt,2);
           double rc = sqrt(ttt_2/(ttt_2 + n - 2));
           double zr = atanh(r) + r / (2 * (n - 1));
           double zrc = atanh(rc);
           double power = stats::pnorm((zr - zrc) * sqrt(n - 3), 0.0, 1) +
                          stats::pnorm((-zr - zrc) * sqrt(n - 3), 0.0, 1);

           // If the calculated power is >= the expected power then we
           // can keep this cluster.  We keep it by adding the correlation
           // and the labels from the original cluster into new variables
           // for the correlation and labels.
           if (power >= _base->_powerThresholdPower) {
             new_correlations.append(correlations[k]);
             new_labels = labels;
             num_final_K++;
           }
       }

       // Prepare the workblock results by adding a new pair. A pair
       // gets added regardless if there are any clusters in it.
       Pair pair;
       pair.K = num_final_K;
       if (num_final_K > 0) {
           pair.correlations = new_correlations;
           pair.labels = new_labels;
       }
       resultBlock->append(pair);

       // Move to the location in the CCM/CMX matrix where the work block starts.
       cmxPair.readNext();
       ccmPair.read(cmxPair.index());
   }

   // We're done! Return the result block.
   return unique_ptr<EAbstractAnalyticBlock>(resultBlock);
}

