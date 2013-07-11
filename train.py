from datetime import datetime
import time
import pickle
import analyzer
import os

if __name__ == '__main__':
    classifier = analyzer.NeuralNetworkClassifier
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
                    #print "Result mask:", data
                    results = list(data)
                    results = [[item] for item in results]
                trainer.add(base + ".wav", results)
                print
                count += 1

    # Train on the loaded dataset
    if count > 0:
        print "Training"
        trainer.train()
    else:
        print "No recordings found"

    # Save classifier
    # filename = "classifier" + str(int(time.time())) + ".xml"
    filename = "classifier.xml"
    trainer.classifier.export(filename)

    print
    print "Saved trained network to:", filename
    print
    print "Finished in", datetime.now() - start_time
