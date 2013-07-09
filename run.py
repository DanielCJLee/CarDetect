from datetime import datetime
import sys
import analyzer


if __name__ == '__main__':
    filename = sys.argv[1]

    print "Processing file", filename

    classifier = analyzer.NeuralNetworkClassifier(analyzer.feature_vector_length(), 1)
    trainer = analyzer.BatchFileTrainer(classifier)
    start_time = datetime.now()

    trainer.add(filename, [])

    print "Finished in", datetime.now() - start_time, "seconds"
