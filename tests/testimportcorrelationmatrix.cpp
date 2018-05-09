#include <ace/core/AceCore.h>
#include <ace/core/datamanager.h>
#include <ace/core/datareference.h>

#include "testimportcorrelationmatrix.h"
#include "analyticfactory.h"
#include "datafactory.h"
#include "importcorrelationmatrix.h"



void TestImportCorrelationMatrix::test()
{
	// create random correlation data
	int numGenes = 10;
	int numSamples = 5;
	int maxClusters = 5;
	QVector<Pair> testPairs;

	for ( int i = 0; i < numGenes; ++i )
	{
		for ( int j = 0; j < i; ++j )
		{
			int numClusters = rand() % (maxClusters + 1);

			if ( numClusters > 0 )
			{
				QVector<QVector<qint8>> clusters(numClusters);
				QVector<float> correlations(numClusters);

				for ( int k = 0; k < numClusters; ++k )
				{
					clusters[k].resize(numSamples);

					for ( int n = 0; n < numSamples; ++n )
					{
						clusters[k][n] = rand() % 2;
					}

					correlations[k] = -1.0 + 2.0 * rand() / (1 << 31);
				}

				testPairs.append({ { i, j }, clusters, correlations });
			}
		}
	}

	// initialize temp files
	QString txtPath {QDir::tempPath() + "/test.txt"};
	QString ccmPath {QDir::tempPath() + "/test.ccm"};
	QString cmxPath {QDir::tempPath() + "/test.cmx"};

	QFile(txtPath).remove();
	QFile(ccmPath).remove();
	QFile(cmxPath).remove();

	// create raw text file
	QFile file(txtPath);
	Q_ASSERT(file.open(QIODevice::ReadWrite));

	QTextStream stream(&file);

	for ( auto& testPair : testPairs )
	{
		for ( int k = 0; k < testPair.clusters.size(); k++ )
		{
			int numSamples = 0;
			int numMissing = 0;
			int numPostOutliers = 0;
			int numPreOutliers = 0;
			int numThreshold = 0;
			QString sampleMask(numSamples);

			for ( int i = 0; i < numSamples; i++ )
			{
				sampleMask[i] = '0' + testPair.clusters[k][i];
			}

			stream
				<< testPair.index.getX()
				<< "\t" << testPair.index.getY()
				<< "\t" << k
				<< "\t" << testPair.clusters.size()
				<< "\t" << numSamples
				<< "\t" << numMissing
				<< "\t" << numPostOutliers
				<< "\t" << numPreOutliers
				<< "\t" << numThreshold
				<< "\t" << testPair.correlations[k]
				<< "\t" << sampleMask
				<< "\n";
		}
	}

	file.close();

	// create analytic object
	EAbstractAnalyticFactory& factory {EAbstractAnalyticFactory::getInstance()};
	std::unique_ptr<EAbstractAnalytic> analytic {factory.make(AnalyticFactory::ImportCorrelationMatrixType)};

	analytic->addFileIn(ImportCorrelationMatrix::InputFile, txtPath);
	analytic->addDataOut(ImportCorrelationMatrix::ClusterData, ccmPath, DataFactory::CCMatrixType);
	analytic->addDataOut(ImportCorrelationMatrix::CorrelationData, cmxPath, DataFactory::CorrelationMatrixType);
	analytic->setArgument(ImportCorrelationMatrix::GeneSize, numGenes);
	analytic->setArgument(ImportCorrelationMatrix::SampleSize, numSamples);
	analytic->setArgument(ImportCorrelationMatrix::CorrelationName, "test");

	// run analytic
	analytic->run();

	// TODO: read and verify cluster data
	// TODO: read and verify correlation data
}