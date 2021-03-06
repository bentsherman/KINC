#include "conditionaltest_serial.h"
#include "conditionaltest_workblock.h"
#include "conditionaltest_resultblock.h"
#include "ccmatrix_pair.h"
#include "correlationmatrix_pair.h"
#include "correlationmatrix.h"
#include "correlationmatrix_pair.h"
#include "expressionmatrix_gene.h"
#include <ace/core/elog.h>
#include <gsl/gsl_fit.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_multifit.h>
#include <gsl/gsl_math.h>



/*!
 * Create a serial object.
 */
ConditionalTest::Serial::Serial(ConditionalTest* parent) : EAbstractAnalyticSerial(parent), _base(parent)
{
    EDEBUG_FUNC(this,parent);
}



/*!
 * Perform the desired work on a result block.
 *
 * @param block The work block you are working on.
 *
 * @return The populated result block.
 */
std::unique_ptr<EAbstractAnalyticBlock> ConditionalTest::Serial::execute(const EAbstractAnalyticBlock* block)
{
    EDEBUG_FUNC(this, block);

    // crate the work and result blocks
    const WorkBlock* workBlock {block->cast<WorkBlock>()};
    ResultBlock* resultBlock {new ResultBlock(workBlock->index(), _base->_numTests, workBlock->startpair())};

    // Create iterators for the CCM data object.
    CCMatrix::Pair ccmPair = CCMatrix::Pair(_base->_ccm);
    CorrelationMatrix::Pair cmxPair = CorrelationMatrix::Pair(_base->_cmx);

    // Iterate through the pairs the workblock is assigned to.
    qint64 start = workBlock->startpair();
    qint64 size = workBlock->size();
    Pairwise::Index index(workBlock->start());

    // iterate through each pair in the matrix
    for ( qint64 ccmIndex = start; ccmIndex < start + size; ccmIndex++ )
    {
        // reads the first value in the ccm
        if ( ccmIndex == start )
        {
            ccmPair.read(index);
            cmxPair.read(ccmPair.index());
        }

        if ( (ccmPair.index().getX() != 1 && ccmPair.index().getY()!= 0) || ccmIndex != start )
        {
            ccmPair.readNext();
            cmxPair.read(ccmPair.index());
        }

        // if the first one isnt in the cluster we should not count it.
        if ( ccmPair.clusterSize() == 0 )
        {
            size++;
            continue;
        }

        // Initialize new pvalues, one set of pvalues for each cluster.
        QVector<QVector<double>> pValues;
        pValues.resize(ccmPair.clusterSize());

        // Initialize new r2, one set of pvalues for each cluster.
        QVector<QVector<double>> r2;
        r2.resize(ccmPair.clusterSize());

        // for each cluster in the pair, run the binomial and linear regression tests.
        for ( qint32 clusterIndex = 0; clusterIndex < ccmPair.clusterSize(); clusterIndex++ )
        {
            // resize for room for each test.
            pValues[clusterIndex].resize(_base->_numTests);
            r2[clusterIndex].resize(_base->_numTests);

            for ( qint32 featureIndex = 0, testIndex = 0; featureIndex < _base->_features.size(); featureIndex++ )
            {
                if ( _base->_testType.at(featureIndex) == NONE || _base->_testType.at(featureIndex) == UNKNOWN )
                {
                    continue;
                }

                if ( _base->_testType.at(featureIndex) == QUANTITATIVE || _base->_testType.at(featureIndex) == ORDINAL )
                {
                    prepAnxData(_base->_features.at(featureIndex).at(0), featureIndex, _base->_testType.at(featureIndex));
                    test(ccmPair, clusterIndex, testIndex, featureIndex, 0, pValues, r2);
                }
                else if ( _base->_testType.at(featureIndex) == CATEGORICAL )
                {
                    for ( qint32 labelIndex = 0; labelIndex < _base->_features.at(featureIndex).size(); labelIndex++ )
                    {
                        prepAnxData(_base->_features.at(featureIndex).at(labelIndex), featureIndex, _base->_testType.at(featureIndex));

                        // if there are sub labels to test for the feature
                        if ( _base->_features.at(featureIndex).size() > 1 )
                        {
                            if ( labelIndex == 0 )
                            {
                                labelIndex = 1;
                            }
                            test(ccmPair, clusterIndex, testIndex, featureIndex, labelIndex, pValues, r2);
                        }

                        // if only the feature needs testing (no sub labels)
                        else
                        {
                            test(ccmPair, clusterIndex, testIndex, featureIndex, 0, pValues, r2);
                        }
                    }
                }
            }
        }

        if ( !isEmpty(pValues) )
        {
            CSMPair pair;
            pair.pValues = pValues;
            pair.r2 = r2;
            pair.x_index = ccmPair.index().getX();
            pair.y_index = ccmPair.index().getY();
            resultBlock->append(pair);
        }
    }
    return std::unique_ptr<EAbstractAnalyticBlock>(resultBlock);
}



/*!
 * Prepare the annotation matrix data for testing.
 *
 * @param testLabel The label you are testing on.
 *
 * @param dataIndex The feature the label is part of.
 *
 * @return The number of samples in total of the test label.
 */
int ConditionalTest::Serial::prepAnxData(QString testLabel, int dataIndex, TESTTYPE testType)
{
    EDEBUG_FUNC(this, testLabel, dataIndex);

    // get the needed data fro the comparison
    _catCount = 0;
    _amxData.resize(_base->_data.at(dataIndex).size());

    // populate array with annotation data
    for ( int j = 0; j < _base->_data.at(dataIndex).size(); j++ )
    {
        _amxData[j] = _base->_data.at(dataIndex).at(j).toString();

        // if data is the same as the test label add one to the catagory counter
        if ( testType == _base->CATEGORICAL && _amxData[j] == testLabel )
        {
            _catCount++;
        }
    }

    return _catCount;
}



/*!
 * Check to see if a matrix is empty.
 *
 * @param vector The matrix you want to check.
 *
 * @return True if the matrix is empty, false otherwise.
 */
bool ConditionalTest::Serial::isEmpty(QVector<QVector<double>>& matrix)
{
    EDEBUG_FUNC(this, &matrix);

    int index = 0;

    while ( index < matrix.size() )
    {
        if ( !matrix.at(index++).isEmpty() )
        {
            return false;
        }
    }

    return true;
}



/*!
 * Prepare the cluster category count information.
 *
 * @param ccmPair The gene pair that we are counting the labels for.
 *
 * @param clusterIndex The number cluster we are in in the pair
 *
 * @return The number of labels in the given cluster.
 */
int ConditionalTest::Serial::clusterInfo(CCMatrix::Pair& ccmPair, int clusterIndex, QString label, TESTTYPE testType)
{
    _catCount = _clusterSize = _catInCluster = 0;

    // Look through all the samples in the mask.
    for ( qint32 i = 0; i < _base->_emx->sampleSize(); i++ )
    {
        // If the sample label matches with the given label.
        if ( testType == _base->CATEGORICAL && _amxData.at(i) == label )
        {
            _catCount++;
        }

        if ( ccmPair.at(clusterIndex, i) == 1 )
        {
            _clusterSize++;

            if ( testType == _base->CATEGORICAL && _amxData.at(i) == label )
            {
                _catInCluster++;
            }
        }
    }

    return _catInCluster;
}



/*!
 * An interface to choose and run the correct tests on the pair.
 *
 * @param ccmPair The pair thats going to be tested.
 *
 * @param clusterIndex The cluster to test inside the pair, each cluster is
 *       tested.
 *
 * @param testIndex Which test we are currently performing.
 *
 * @param featureIndex The current feature we are testing.
 *
 * @param labelIndex The label in the feature we are running a test on.
 *
 * @param pValues The two dimensional array holding all of the results from the
 *       tests.
 *
 * @return The test that was just conducted.
 */
int ConditionalTest::Serial::test(
    CCMatrix::Pair& ccmPair,
    qint32 clusterIndex,
    qint32& testIndex,
    qint32 featureIndex,
    qint32 labelIndex,
    QVector<QVector<double>>& pValues,
    QVector<QVector<double>>& r2)
{
    EDEBUG_FUNC(this,&ccmPair, clusterIndex, &testIndex, featureIndex, labelIndex, &pValues);

    // get informatiopn on the mask
    clusterInfo(ccmPair, clusterIndex, _base->_features.at(featureIndex).at(labelIndex), _base->_testType.at(featureIndex));

    // For linear regresssion we need a variable that will hold the
    // pvalue and the r2 value.
    QVector<double> results(2);

    // conduct the correct test based on the type of data
    switch(_base->_testType.at(featureIndex))
    {
        case CATEGORICAL:
            pValues[clusterIndex][testIndex] = hypergeom(ccmPair, clusterIndex,
                                                         _base->_features.at(featureIndex).at(labelIndex));
            r2[clusterIndex][testIndex] = qQNaN();
            testIndex++;
            break;
        case ORDINAL:
            regression(_amxData, ccmPair, clusterIndex, ORDINAL, results);
            pValues[clusterIndex][testIndex] = results.at(0);
            r2[clusterIndex][testIndex] = results.at(1);
            testIndex++;
            break;
        case QUANTITATIVE:
            regression(_amxData, ccmPair, clusterIndex, QUANTITATIVE, results);
            pValues[clusterIndex][testIndex] = results.at(0);
            r2[clusterIndex][testIndex] = results.at(1);
            testIndex++;
            break;
        default:
            break;
    }

    return _base->_testType.at(featureIndex);
}



/*!
 * Run the first binomial test for given data.
 *
 * @return Pvalue corrosponding to the test.
 */
double ConditionalTest::Serial::hypergeom(CCMatrix::Pair& ccmPair, int clusterIndex, QString test_label)
{
    EDEBUG_FUNC(this);

    // We use the hypergeometric distribution because the samples are
    // selected from the population for membership in the cluster without
    // replacement.

    // If a population contains n_1 elements of “type 1” and n_2 elements of
    // “type 2” then the hypergeometric distribution gives the probability
    // of obtaining k elements of “type 1” in t samples from the population
    // without replacement.

    int sampleSize =  _base->_emx->sampleSize();

    // Population contains n1 elements of Type 1.
    int n1 = _catCount;
    // Population contains n2 elements of Type 2.
    int n2 = sampleSize- _catCount;
    // k elements of Type 1 were selected.
    int k = _catInCluster;
    // t total elements were selected.
    int t = _clusterSize;

    // If n1 == k we will always get a zero because we've
    // reached the end of the distribution, so the Ho that
    // X > k is always 0.  This happens if the cluster is 100%
    // comprised of the category we're looking for.
    if ( k == n1 )
    {
        return 1;
    }

    // If our dataset is large, the power to detect the effect
    // size increases, resulting in potentially insignificant
    // proportions having signficant p-values. Using Cohen's H
    // to set a large-effect size (e.g. 0.8) with a sig.level of
    // 0.001 and a power of 0.95 we need at least 31 samples.
    // So, we'll perform a jacknife resampling of our data
    // to calculate an average proportion of 31 samples
    if ( t > 31 )
    {
        // Initialize the uniform random number generator.
        const gsl_rng_type * T;
        gsl_rng * r;
        gsl_rng_env_setup();
        T = gsl_rng_default;
        r = gsl_rng_alloc (T);

        // Holds the jacknife average proportion.
        int jkap = 0;

        // To perform the Jacknife resampling we will
        // perform 30 iterations (central limit thereom)
        int in = 30;
        for ( int i = 0; i < in; i++ )
        {
            // Keeps track of the number of successes for each iteration.
            int ns = 0;

            // Generate 31 random indexes between 0 and the size of
            // the sample string.  We will use these numbers to
            // randomly select a sample and if it is a 1 and of the
            // testing category then we consider it a success.
            int indexes[sampleSize];
            int chosen[31];

            for ( int j = 0; j < sampleSize; j++ )
            {
                indexes[j] = j;
            }

            // The gsl_ran_choose function randmly choose samples without replacement from a list.
            gsl_ran_choose(r, chosen, 31, indexes, sampleSize, sizeof(int));

            for ( int j = 0; j < 31; j++ )
            {
                if ( ccmPair.at(clusterIndex, chosen[j]) == 1 && _amxData.at(chosen[j]) == test_label )
                {
                    ns = ns + 1;
                }
            }

            jkap += ns;
        }

        // Calculate the average proportion from all iterations
        // and free the random number struct.
        jkap = jkap/in;
        gsl_rng_free(r);

        // Now reset the sample size and the proporiton of success.
        k = jkap;
        t = 31;
    }

    // The gsl_cdf_hypergeometric_Q function uses the upper-tail of the CDF.
    double pvalue = gsl_cdf_hypergeometric_Q(k, n1, n2, t);
    return pvalue;
}



/*!
 * Run the regression test for given data, the regression line is genex vs geney vs label data.
 *
 * @param amxInfo Annotation matrix information.
 *
 * @param ccmPair Cluster matrix pair.
 *
 * @param clusterIndex The index of the cluster, used to get the right info
 *       from the cluster pair
 *
 * @return Pvalue corrosponding to the test.
 */
void ConditionalTest::Serial::regression(QVector<QString> &amxInfo, CCMatrix::Pair& ccmPair, int clusterIndex, TESTTYPE testType, QVector<double>& results)
{
    EDEBUG_FUNC(this, &amxInfo, &ccmPair, clusterIndex);

    // Temp containers.
    QVector<double> labelInfo;

    // Regression model containers.
    double chisq;
    gsl_matrix *X, *cov;
    gsl_vector *Y, *C;
    double pValue = 0.0;
    double r2 = 0.0;

    // Allocate a matrix to hold the predictior variables, in this case the gene
    // Expression data.
    X = gsl_matrix_alloc(_clusterSize, 4);

    // Allocate a vector to hold observation data, in this case the data
    // corrosponding to the features.
    Y = gsl_vector_alloc(_clusterSize);

    // Allocate a vector and matrix for the slope info.
    C = gsl_vector_alloc(4);
    cov = gsl_matrix_alloc(4, 4);

    // Read in the gene pairs expression information.
    ExpressionMatrix::Gene geneX(_base->_emx);
    ExpressionMatrix::Gene geneY(_base->_emx);
    geneX.read(ccmPair.index().getX());
    geneY.read(ccmPair.index().getY());

    // Look through all the samples in the mask.
    for ( int i = 0, j = 0; i < _base->_emx->sampleSize(); i++ )
    {
        // If the sample label matches with the given label.
        if ( ccmPair.at(clusterIndex, i) == 1 )
        {
            // Add emx data as the predictors but only for samples in the cluster. We
            // add a 1 for the intercept, gene1, gene2 and the interaction: gene1*gene2.
            // The interaction term handles the case where the relationship is dependent
            // on the values of both genes.
            double g1 = static_cast<double>(geneX.at(i));
            double g2 = static_cast<double>(geneY.at(i));

            gsl_matrix_set(X, j, 0, static_cast<double>(1)); // for the intercept
            gsl_matrix_set(X, j, 1, g1);
            gsl_matrix_set(X, j, 2, g2);
            gsl_matrix_set(X, j, 3, g1*g2);

            // Next add the annotation observation for this sample to the Y vector.
            if ( testType == ORDINAL )
            {
                // Convert the observation data into a "design vector"
                // Each unique number being a ssigned a unique integer.
                if ( !labelInfo.contains(amxInfo.at(i).toInt()) )
                {
                    labelInfo.append(amxInfo.at(i).toInt());
                }

                for ( int k = 0; k < labelInfo.size(); k++ )
                {
                    if ( labelInfo.at(k) == amxInfo.at(i).toInt() )
                    {
                        gsl_vector_set(Y, j, k + 1);
                    }
                }
            }
            else
            {
                gsl_vector_set(Y, j, amxInfo.at(i).toFloat());
            }

            j++;
        }
    }

    // Create the workspace for the gnu scientific library to work in.
    gsl_multifit_linear_workspace * work = gsl_multifit_linear_alloc (X->size1, X->size2);

    // Regrassion calculation.
    gsl_multifit_linear(X, Y, C, cov, &chisq, work);

    // Calculate R^2 and p-value
    r2 = 1 - chisq / gsl_stats_tss(Y->data, Y->stride, Y->size);

    double dl = _clusterSize - 2;
    double F = r2 * dl / (1 - r2);

    pValue = 1 - gsl_cdf_fdist_P (F, 1, dl);

    // TODO: we should check the assumptions of the linear regression and
    // not return if the assumptions are not met.

    // TODO: it would be nice to return a rate of change of the conditioal mean.

    // TODO: it would be nice to return the r2 value along with the p-value.

    // Four scenarios:
    // 1) low R-square and low p-value (p-value <= 0.05).  Model doesn't explain
    //    the variation but it does follow the trend or regression line well.
    // 2) low R-square and high p-value (p-value > 0.05).  Model doesn't explai
    //    the variation and it doesn't follow the trend line.
    // 3) high R-square and low p-value.  Model explains the variance it
    //    follows the trend line.
    // 4) high R-square and high p-value. Model explains the variance well but
    //    it does not follow the trend line very well.
    // In summary, low p-values still indicate a real relationship between the
    // predictors and the observed values.

    // Free all of the data.
    gsl_matrix_free(X);
    gsl_vector_free(Y);
    gsl_matrix_free(cov);
    gsl_vector_free(C);
    gsl_multifit_linear_free(work);

    // Set the results array
    if ( qIsNaN(pValue) )
    {
        results[0] = 1;
        results[1] = 0;
    }
    else {
        results[0] = pValue;
        results[1] = r2;
    }
}
