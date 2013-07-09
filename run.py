from datetime import datetime
import sys
import analyzer
import pickle

if __name__ == '__main__':
    filename = sys.argv[1]
    classifier_filename = sys.argv[2]

    print "Loading classifier:", classifier_filename
    classifier = pickle.load(open(classifier_filename, 'rb'))
    file_analyzer = analyzer.FileAnalyzer(classifier)
    start_time = datetime.now()

    print "Processing file:", filename
    print file_analyzer.analyze(filename)

    print "Finished in", datetime.now() - start_time
