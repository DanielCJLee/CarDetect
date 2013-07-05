from datetime import datetime
import sys
from scipy.io import wavfile
from analyzer import Analyzer

BUFFER_SIZE = 1000


def virtual_buffer(data, length):
    for i in xrange(0, len(data), length):
        yield data[i:i+length]


if __name__ == '__main__':
    filename = sys.argv[1]
    print "Processing file", filename
    rate, data = wavfile.read(filename)
    analyzer = Analyzer(rate)
    start_time = datetime.now()
    for samples in virtual_buffer(data, BUFFER_SIZE):
        analyzer.push(samples)
    print "Finished in", datetime.now() - start_time, "seconds"
    analyzer.display()