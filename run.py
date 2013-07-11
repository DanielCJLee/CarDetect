from datetime import datetime
import sys
import analyzer
import pickle

if __name__ == '__main__':
    filename = sys.argv[1]
    # classifier_filename = sys.argv[2]
    classifier_filename = "classifier.xml"

    print "Loading classifier:", classifier_filename
    classifier = analyzer.SavedNeuralNetworkClassifier(classifier_filename)
    file_analyzer = analyzer.FileAnalyzer(classifier)
    start_time = datetime.now()

    print "Processing file:", filename
    print ", ".join([str(item) for item in file_analyzer.analyze(filename)])

    print "Finished in", datetime.now() - start_time
