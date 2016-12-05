#ifndef _MIXMODPWCLUSTERS_
#define _MIXMODPWCLUSTERS_

#include <mixmod/Kernel/IO/Data.h>
#include <mixmod/Kernel/IO/Label.h>
#include <mixmod/Kernel/IO/LabelDescription.h>
#include <mixmod/Kernel/IO/GaussianData.h>
#include <mixmod/Kernel/IO/DataDescription.h>
#include <mixmod/Clustering/ClusteringInput.h>
#include <mixmod/Clustering/ClusteringOutput.h>
#include <mixmod/Clustering/ClusteringModelOutput.h>
#include <mixmod/Clustering/ClusteringMain.h>
#include <mixmod/Clustering/ClusteringStrategy.h>

#include "PairWiseCluster.h"
#include "PairWiseClustering.h"
#include "PairWiseClusterList.h"
#include "../../stats/outlier.h"

/**
 * A class for a single pair-wise comparision using mixture models.
 *
 * This class receives via it's constructor the PairWiseSet object that
 * contains the information about the two genes whose samples will be
 * clustered.
 */
class MixtureModelPWClusters {
  private:
    // The list of clusters generated by this clustering method.
    PairWiseClusterList * pwcl;
    // The pair of genes on which the clustering will occur.
    PairWiseSet * pwset;
    // The MixMod Lib wants the data as an n x 2 array of doubles.  So
    // the construct will extract the data from the pwset object and
    // convert it into this data array.
    double ** data;
    // The vector containing the cluster membership.
    int64_t * labels;
    // The criterion model. E.g.  BIC, ICL, NEC, CV, DCV.
    char criterion[4];
    // The maximum number of clusters to allow per comparision.
    int min_obs;
    // The similarity method
    char ** method;
    // The number of similarity methods
    int num_methods;

  public:
    MixtureModelPWClusters(PairWiseSet *pwset, int min_obs,
        char ** method, int num_methods);
    ~MixtureModelPWClusters();

    // Returns the criterion used for mixture model.
    char * getCriterion() { return criterion; }
    // Returns the array of numeric cluster labels assigned to each sample.
    int64_t * getLabels() { return labels; }
    PairWiseClusterList * getClusterList() { return pwcl; };

    // Performs pair-wise clustering and similiarity calculation.
    void run(char * criterion, int max_clusters);
};

#endif