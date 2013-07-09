import struct
import analyzer
from datetime import datetime
import pickle
import sys
import pyaudio

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

print "Starting PyAudio input"
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

classifier_filename = sys.argv[1]
print "Loading classifier:", classifier_filename
classifier = pickle.load(open(classifier_filename, 'rb'))

realtime_analyzer = analyzer.RealtimeAnalyzer(RATE, classifier)

print "Now recording"
while True:
    data = stream.read(CHUNK)
    data = struct.unpack("%dh" % CHUNK, data)
    realtime_analyzer.push(data)
