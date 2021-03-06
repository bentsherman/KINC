#include "conditionaltest.h"
#include "conditionaltest_input.h"
#include "conditionaltest_resultblock.h"
#include "conditionaltest_workblock.h"
#include "conditionaltest_serial.h"
#include "conditionspecificclustersmatrix.h"
#include "conditionspecificclustersmatrix_pair.h"
#include "ccmatrix_pair.h"
#include <ace/core/elog.h>
#include <ace/core/ace_qmpi.h>



/*!
 * Supplies ACE with the number of work blocks it is going to create.
 *
 * @return How many pieces you want to break up the task your working on.
 */
int ConditionalTest::size() const
{
    EDEBUG_FUNC(this);

    return (static_cast<qint64>(_ccm->size()) + _workBlockSize - 1) / _workBlockSize;
}



/*!
 * Return the total number of pairs that must be processed for a given
 * expression matrix.
 *
 * @param cmx The cluster matrix for the data.
 *
 * @return The total number of pair to process.
 */
qint64 ConditionalTest::totalPairs(const CorrelationMatrix* cmx)
{
    return static_cast<qint64>(cmx->geneSize()) * (cmx->geneSize() - 1) / 2;
}



/*!
 * An interface that processes result blocks once the serial is done working on
 * them.
 *
 * @param result The processed work block from the serial.
 */
void ConditionalTest::process(const EAbstractAnalyticBlock* result)
{
    EDEBUG_FUNC(this,result);

    if ( ELog::isActive() )
    {
        ELog() << tr("Processing result %1 of %2.\n").arg(result->index()).arg(size());
    }

    if ( result->index() == 0 )
    {
        _out->setTestCount(_numTests);
    }

    // Iterate through the result block pairs.
    const ResultBlock* resultBlock {result->cast<ResultBlock>()};

    for ( qint32 i = 0; i < resultBlock->pairs().size(); i++ )
    {
        // copy the values form the pairs to the CSM
        if ( resultBlock->pairs().at(i).pValues.size() > 0 )
        {
            // Create pair objects for the output data file.
            CSMatrix::Pair CSMPair(_out);
            Pairwise::Index index(resultBlock->pairs().at(i).x_index, resultBlock->pairs().at(i).y_index);
            _index = index;

            // Iterate through the clusters in the pair.
            for (int j = 0; j < resultBlock->pairs().at(i).pValues.size(); ++j )
            {
                // add each cluster into the CSM
                CSMPair.addCluster(1, _numTests);
                for ( int k = 0; k < resultBlock->pairs().at(i).pValues.at(j).size(); k++ )
                {
                    CSMPair.at(j, k, "pvalue") = resultBlock->pairs().at(i).pValues.at(j).at(k);
                    CSMPair.at(j, k, "r2") = resultBlock->pairs().at(i).r2.at(j).at(k);
                }
            }
            // write the info into the CSM
            CSMPair.write(index); // have an error here
        }
    }
}



/*!
 * An interface to create a new input data object.
 *
 * @return Pointer to the new input data object.
 */
EAbstractAnalyticInput* ConditionalTest::makeInput()
{
    EDEBUG_FUNC(this);

    return new Input(this);
}



/*!
 * Creates a block of work at the given index.
 *
 * @param index The index at which the work block should be made.
 *
 * @return Pointer to the work block.
 */
std::unique_ptr<EAbstractAnalyticBlock> ConditionalTest::makeWork(int index) const
{
    EDEBUG_FUNC(this,index);

    if ( ELog::isActive() )
    {
        ELog() << tr("Making work index %1 of %2.\n").arg(index).arg(size());
    }

    qint64 start {index * static_cast<qint64>(_workBlockSize)};
    qint64 size {std::min(_ccm->size() - start, static_cast<qint64>(_workBlockSize))};

    return std::unique_ptr<EAbstractAnalyticBlock>(new WorkBlock(index, _index, start, size));
}



/*!
 * create uninitialized work blocks.
 *
 * @return a pointer to an uninitialized work block
 */
std::unique_ptr<EAbstractAnalyticBlock> ConditionalTest::makeWork() const
{
    EDEBUG_FUNC(this);

    return std::unique_ptr<EAbstractAnalyticBlock>(new WorkBlock());
}



/*!
 * create uninitialized result blocks.
 *
 * @return a pointer to an uninitialized result block
 */
std::unique_ptr<EAbstractAnalyticBlock> ConditionalTest::makeResult() const
{
    EDEBUG_FUNC(this);

    return std::unique_ptr<EAbstractAnalyticBlock>(new EAbstractAnalyticBlock());
}



/*!
 * Create a new serial object.
 *
 * @return Pointer to a serial new object.
 */
EAbstractAnalyticSerial* ConditionalTest::makeSerial()
{
    EDEBUG_FUNC(this);

    return new Serial(this);
}



/*!
 * An interface to initialize the analytic process, it reads in the line number
 * of the annotaion matrix, so we can read in the data properly.
 */
void ConditionalTest::initialize()
{
    EDEBUG_FUNC(this);

    // only the master process needs to validate arguments
    auto& mpi {Ace::QMPI::instance()};

    if ( !mpi.isMaster() )
    {
        return;
    }

    // make sure input data is valid
    if ( !_ccm )
    {
        E_MAKE_EXCEPTION(e);
        e.setTitle(tr("Invalid Argument"));
        e.setDetails(tr("Did not get valid CCM data argument."));
        throw e;
    }

    if ( !_cmx )
    {
        E_MAKE_EXCEPTION(e);
        e.setTitle(tr("Invalid Argument"));
        e.setDetails(tr("Did not get valid CMX data argument."));
        throw e;
    }

    if ( !_amx )
    {
        E_MAKE_EXCEPTION(e);
        e.setTitle(tr("Invalid Argument"));
        e.setDetails(tr("Did not get valid AMX data argument."));
        throw e;
    }

    if (  !_emx )
    {
        E_MAKE_EXCEPTION(e);
        e.setTitle(tr("Invalid Argument"));
        e.setDetails(tr("Did not get valid EMX data argument."));
        throw e;
    }

    if (  !_out)
    {
        E_MAKE_EXCEPTION(e);
        e.setTitle(tr("Invalid Argument"));
        e.setDetails(tr("Did not get valid output file argument."));
        throw e;
    }

    // open the stream to the coprrect file.
    _stream.setDevice(_amx);

    // atain number of lines in the file.
    while ( !_stream.atEnd() )
    {
        _stream.readLine();
        _amxNumLines++;
    }

    // go back to the begginning
    _stream.seek(0);

    // make sure reading input file worked
    if ( _stream.status() != QTextStream::Ok )
    {
        E_MAKE_EXCEPTION(e);
        e.setTitle(tr("File IO Error"));
        e.setDetails(tr("Qt Text Stream encountered an unknown error."));
        throw e;
    }

    // read in the annotation matrix, as it holds the details on what we need to do for our tests
    Test();
    override();
    readInAMX(_features, _data, _testType);

    rearrangeSamples();

    // initialize work block size
    if ( _workBlockSize == 0 )
    {
        int numWorkers = std::max(1, mpi.size() - 1);

        _workBlockSize = std::min(32768LL, _ccm->size() / numWorkers);
    }

    // CSM specific
    qint32 maxCluster = 64, subHeadersize = 12;
    initialize(maxCluster,subHeadersize,_features,_testType,_data);
}



/*!
 * Validate output arguments for this analytic.
 */
void ConditionalTest::initializeOutputs()
{
    EDEBUG_FUNC(this);

    if ( !_out )
    {
        E_MAKE_EXCEPTION(e);
        e.setTitle(tr("Invalid Argument"));
        e.setDetails(tr("The required output data object was not set."));
        throw e;
    }
}



/*!
 * An interface to read in the contents of an annotation matrix and produces
 * useful information from it.
 *
 * @param amxdata An initially empty array, stores the features and labels of
 *       the annotation matrix.
 *
 * @param data An initially empty array, stores all of the data corrosponding
 *       to the features in the annotation matrix.
 *
 * @param dataTestType The type of test we will run on the data under a
 *       particular feature.
 */
void ConditionalTest::readInAMX(
    QVector<QVector<QString>>& amxdata,
    QVector<QVector<QVariant>>& data,
    QVector<TESTTYPE>& dataTestType)
{
    EDEBUG_FUNC(this,&amxdata,&data,&dataTestType);

    // set the stream to the right file
    _stream.setDevice(_amx);
    _stream.seek(0);

    // read a line form the input file
    QString line = _stream.readLine();

    // the file can be a csv or a tab delimited AMX
    // default is tab diliniated
    if ( _delimiter == "tab" )
    {
        _delimiter = "\t";
    }

    // splits the file along the commas or tabs depending on what it has inside
    auto words = line.split(_delimiter, QString::SkipEmptyParts, Qt::CaseInsensitive);

    // If a field never changes, we don't have to store copies of that data.
    QVector<int> changed;

    // put the column names into the amxdata array, that will be later put into the meta data
    for ( int i = 0; i < words.size(); i++ )
    {
        amxdata.append(QVector<QString>());
        amxdata[i].append(words[i]);
        dataTestType.append(UNKNOWN);
        data.append(QVector<QVariant>());
        changed.append(1);
    }

    configureTests(dataTestType);

    for ( int i = 1; i < _amxNumLines; i++ )
    {
        line = _stream.readLine();
        // splits the file along the commas
        auto words2 = line.split(_delimiter, QString::KeepEmptyParts, Qt::CaseInsensitive);

        // add the data to our arrays
        for ( int j = 0; j < words2.size(); j++ )
        {
            data[j].append(words2[j]);
            if ( dataTestType.at(j) == CATEGORICAL )
            {
                // this will add treatments types, leaf types, and any other types into the meta data
                if ( !amxdata.at(j).contains(words2[j]) )
                {
                    amxdata[j].append(words2[j]);
                    changed[j] = 0;
                }
            }
        }
    }

    // we need to change the test types here if the user has overridden them
    for ( int i = 0; i < _override.size(); i++ )
    {
        for ( int j = 0; j < dataTestType.size(); j++ )
        {
            if ( amxdata.at(j).at(0) == _override.at(i).at(0) )
            {
                if ( QString::compare(_override.at(i).at(1), "CATEGORICAL", Qt::CaseInsensitive) == 0 )
                {
                    dataTestType[j] = CATEGORICAL;
                }
                else if ( QString::compare(_override.at(i).at(1), "ORDINAL", Qt::CaseInsensitive) == 0 )
                {
                    dataTestType[j] = ORDINAL;
                }
                else if ( QString::compare(_override.at(i).at(1), "QUANTITATIVE", Qt::CaseInsensitive) == 0 )
                {
                    dataTestType[j] = QUANTITATIVE;
                }
                else {
                    E_MAKE_EXCEPTION(e);
                    e.setTitle(tr("Unknown override type in the --feat_type argument."));
                    e.setDetails(tr("Unknown override type: %1.  Valid choices are: %2")
                        .arg(_override.at(i).at(1))
                        .arg("categorical, quantitative, ordinal"));
                    throw e;
                }
            }
        }
    }

    // Add in labels the user has chosen to test.
    for ( int j = 0; j < amxdata.size(); j++ )
    {
        int check = 0;
        for ( int k = 0; k < _Test.size(); k++ )
        {
            if ( amxdata.at(j).at(0) == _Test.at(k) )
            {
                check = 1;
            }
        }
        if ( check == 0 )
        {
            dataTestType[j] = NONE;
        }
    }
}



/*!
 * An interface to decide what the test types are going to be for each feature.
 *
 * @param dataTestType An array storing the test information for each feature.
 */
void ConditionalTest::configureTests(QVector<TESTTYPE>& dataTestType)
{
    EDEBUG_FUNC(this,&dataTestType);

    // Counts here for an aproximation for what test type it should be.
    QVector<QVector<qint32>> counts;
    counts.resize(dataTestType.size());

    for ( int i = 0 ; i < counts.size(); i++ )
    {
        counts[i].resize(3);
    }

    bool ok;
    QString line;

    for ( int i = 0; i < 10; i++ )
    {
        // read in the line
        line = _stream.readLine();

        // splits the file along the commas or tabs depending
        if ( _delimiter == "tab" )
        {
            _delimiter = "\t";
        }

        auto words = line.split(_delimiter, QString::SkipEmptyParts, Qt::CaseInsensitive);

        for ( int i = 0; i < words.size(); i++ )
        {
            // it most likeley a double and should be considered QUANTITATIVE.
            // will look for (at laest one number)(a '.')(at laest one number)
            if ( words[i].contains(QRegExp("\\d+\\.\\d+")) )
            {
                if ( i < dataTestType.size() )
                {
                    counts[i][QUANTITATIVE]++;
                }
            }

            // it is most likely an integer and should ne considered Ordinal.
            else if ( words[i].toInt(&ok) && ok )
            {
                if ( i < dataTestType.size() )
                {
                    counts[i][ORDINAL]++;
                }
            }

            // if its neistartingPointther on of those than its a string.
            else
            {
                if ( i < dataTestType.size() )
                {
                    counts[i][CATEGORICAL]++;
                }
            }
        }
    }

    for ( int i = 0; i < counts.size() ; i++ )
    {
        if ( counts.at(i).at(QUANTITATIVE) > 0)
        {
            dataTestType[i] = QUANTITATIVE;
        }
        else if ( counts.at(i).at(ORDINAL) > 0 )
        {
            dataTestType[i] = ORDINAL;
        }
        else
        {
            dataTestType[i] = CATEGORICAL;
        }
    }

    _stream.seek(0);
    _stream.readLine();
}



/*!
 * An interface to find the max index of a vector.
 *
 * @param counts Vector to find the max index of.
 *
 * @return The index of the maximum balue in the vector.
 */
int ConditionalTest::max(QVector<qint32> &counts) const
{
    EDEBUG_FUNC(this,&counts);

    int maxnum = 0;

    if ( counts.size() > 0 )
    {
        qint32 max = counts.at(0);

        for ( int i = 1; i < counts.size(); i++ )
        {
            if ( counts.at(i) > max )
            {
                max = counts.at(i);
                maxnum = i;
            }
        }
    }

    return maxnum;
}



/*!
 * An interface to sperate the test out.
 */
void ConditionalTest::Test()
{
    EDEBUG_FUNC(this);

    // make sure input data is valid
    _Test = _Testing.split(",", QString::SkipEmptyParts, Qt::CaseInsensitive).toVector();

    if ( _Test.isEmpty() )
    {
        E_MAKE_EXCEPTION(e);
        e.setTitle(tr("Invalid Argument"));
        e.setDetails(tr("Please provide Features to Test"));
        throw e;
    }
}



/*!
 * An interface to override testing types.
 */
void ConditionalTest::override()
{
    EDEBUG_FUNC(this);

    auto words = _testOverride.split(",", QString::SkipEmptyParts, Qt::CaseInsensitive).toVector();

    for ( int i = 0; i < words.size(); i++ )
    {
        _override.append(QVector<QString>());

        auto word = words.at(i).split(":");

        for ( auto item : word )
        {
            _override[i].append(item);
        }
    }
}



/*!
 * An interface to provide the names for the tests, creating an easier way to
 * label the data in the ouptut file.
 */
QString ConditionalTest::testNames()
{
    QString string;

    for ( int i = 0; i < _features.size(); i++ )
    {
        if ( _testType.at(i) == CATEGORICAL )
        {
            for ( int j = 1; j < _features.at(i).size(); j++ )
            {
                string += _features.at(i).at(0);
                string += "__";
                string += _features.at(i).at(j);
                string += ":";
            }
        }
        else if ( _testType.at(i) == QUANTITATIVE || _testType.at(i) == ORDINAL )
        {
            string += _features.at(i).at(0);
            string += ":";
        }
    }

    return string;
}



/*!
 * An interface to initialize the metadata corrosponding to the data retrived
 * from the annotation matrix.
 *
 * @param maxClusterSize The maximum number of clusters that can be stored in
 *       a pair.
 *
 * @param subHeaderSize The size of the subheader of the matrix, in this case
 *       it only stores sample size.
 *
 * @param amxData Stores feature names from the annotation array.
 *
 * @param testtype The type of test that was ran on the features.
 *
 * @param genenames Names of all the genes from the gene expression matrix.
 *
 * @param data All of the data corrosponding to the features from the annotation
 *       array.
 */
void ConditionalTest::initialize(qint32 &maxClusterSize, qint32 &subHeaderSize,QVector<QVector<QString>> &amxData, QVector<TESTTYPE> &testType, QVector<QVector<QVariant>> &data)
{
    EDEBUG_FUNC(this,&maxClusterSize,&subHeaderSize,&amxData,&testType,&data);

    // needed by the CSM specific initializer
    EMetaArray Features;
    QVector<EMetaArray> featureInfo;
    QVector<EMetaArray> Data;
    Data.resize(data.size());
    featureInfo.resize(amxData.size());

    // Meta Data for Features.
    for ( int i = 0; i < amxData.size(); i++ )
    {
        Features.append(amxData.at(i).at(0));
    }

    // Meta Data for the test types of the catagories.
    for ( int i = 0; i < featureInfo.size(); i++ )
    {
        switch(testType.at(i))
        {
        case CATEGORICAL : featureInfo[i].append(tr("Categorical"));
            break;
        case ORDINAL : featureInfo[i].append(tr("Ordinal"));
            break;
        case QUANTITATIVE : featureInfo[i].append(tr("Quantitative"));
            break;
        case NONE : featureInfo[i].append(tr("None"));
            break;
        default : featureInfo[i].append(tr("Unknown"));
            break;
        }
    }

    // Meta Data for the sub catagories.
    for ( int i = 0; i < amxData.size(); i++ )
    {
        for ( int j = 0; j < amxData.at(i).size(); j++ )
        {
            featureInfo[i].append(amxData.at(i).at(j));
        }
    }

    // Meta Data for all the data in the annotation matrix.
    for ( int i = 0; i < data.size(); i++ )
    {
        for ( int j = 0; j < data.at(i).size(); j++ )
        {
            Data[i].append(data.at(i).at(j).toString());
        }
    }

    // inserts the Meta Data into the CSM Object stored on the disk.
    _out->initialize(Features, featureInfo, Data, _numTests, testNames());
    _out->initialize(_emx->geneNames(), maxClusterSize, subHeaderSize);
}



/*!
 * An interface to move the amx data around so that the emx samples and the
 * annotation matrix samples are in the same order
 */
void ConditionalTest::rearrangeSamples()
{
    int sampleIndex = 0;
    int startIndex = 0;
    QVariant temp;

    // find the sample index, if none is found with the name "samples" then
    // we default to using the first column.
    for ( int i = 0; i< _features.size(); i++ )
    {
        if ( QString::compare(_features.at(i).at(0), "samples", Qt::CaseInsensitive) ||
             QString::compare(_features.at(i).at(0), "sample", Qt::CaseInsensitive) )
        {
            sampleIndex = i;
            break;
        }
    }

    // if the sample size differes break.
    if ( _data.at(sampleIndex).size() != _emx->sampleSize() )
    {
        E_MAKE_EXCEPTION(e);
        e.setTitle(tr("Sample Size Error"));
        e.setDetails(tr("Sample size in emx does not match annotation matrix."));
        throw e;
    }

    // look through all the samples, if you find one that doesnt match
    // switch it for the right sample.
    for ( int i = 0; i < _emx->sampleSize(); i++ )
    {
        if ( _emx->sampleNames().at(i).toString() != _data.at(sampleIndex).at(i) )
        {
            // find the right sample.
            for ( int j = startIndex; j <  _data.at(sampleIndex).size(); j++ )
            {
                // switch the samples.
                if ( _emx->sampleNames().at(i).toString() == _data.at(sampleIndex).at(j) )
                {
                    for ( int k = 0; k < _data.size(); k++ )
                    {
                        // move wrong sample to temp.
                        temp = _data.at(k).at(i);
                        // move right sample to the right place.
                        _data[k][i] = _data.at(k).at(j);
                        // replace the right sample with the wrong sample.
                        _data[k][j] = temp;
                    }
                    break;
                }

                if ( j == _emx->sampleSize() - 1 )
                {
                    E_MAKE_EXCEPTION(e);
                    e.setTitle(tr("Sample Size Error"));
                    e.setDetails(tr("Sample not in emx."));
                    throw e;
                }
            }

            // increment the start nidex so you dont have to start at 0 every time.
            startIndex++;
        }
    }
}
