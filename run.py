import sys
from scipy.io import wavfile
from analyzer import Analyzer

BUFFER_SIZE = 1000


def virtual_buffer(data, length):
    for i in xrange(0, len(data), length):
        yield data[i:i+length]


if __name__ == '__main__':
    rate, data = wavfile.read(sys.argv[1])
    analyzer = Analyzer(rate)
    for piece in virtual_buffer(data, BUFFER_SIZE):
        analyzer.push(piece)
        analyzer.update()