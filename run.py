from datetime import datetime
import sys
import analyzer
import matplotlib.pyplot as plt
import os
import numpy as np


THRESHOLD = 0.5


if __name__ == '__main__':
    filename = sys.argv[1]
    # classifier_filename = sys.argv[2]
    script_directory = os.path.dirname(os.path.realpath(__file__))
    classifier_filename = os.path.join(script_directory, "classifier.xml")
    save_filename = os.path.join(script_directory, "analysis.csv")

    print "Loading classifier:", classifier_filename
    classifier = analyzer.SavedNeuralNetworkClassifier(classifier_filename)
    file_analyzer = analyzer.FileAnalyzer(classifier)
    start_time = datetime.now()

    print "Processing file:", filename
    results = file_analyzer.analyze(filename, save_filename=save_filename)
    print "Finished in", datetime.now() - start_time

    output = np.array(results).T
    processed_output = [0] * len(output[0])
    for row in output:
        for i in xrange(len(row)):
            if row[i] >= THRESHOLD:
                processed_output[i] += 1
    plt.plot(processed_output)
    plt.show()
