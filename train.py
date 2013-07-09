from datetime import datetime
import analyzer
import os

if __name__ == '__main__':
    classifier = analyzer.NeuralNetworkClassifier(analyzer.feature_vector_length(), 1, n_hidden=2*analyzer.feature_vector_length())
    trainer = analyzer.BatchFileTrainer(classifier)
    start_time = datetime.now()

    # Load all files in the recordings directory, if it has a corresponding results file
    count = 0
    for root, _, files in os.walk('recordings'):
        for f in files:
            if f[-4:] == ".txt":
                print "Processing recording:", f[0:-4]
                base = os.path.join(root, f[0:-4])
                with open(base + ".txt", "r") as x:
                    data = x.readline()
                    print data
                    results = list(data)
                trainer.add(base + ".wav", results)
                count += 1

    # Train on the loaded dataset
    if count > 0:
        print "Training"
        trainer.train()
    else:
        print "No recordings found"

    print "Finished in", datetime.now() - start_time
