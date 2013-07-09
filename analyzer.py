"""
Created on Jul 3, 2013

@author: Zachary
"""
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import mlab
from pybrain.supervised import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from scipy.io import wavfile
import inspect
from datetime import datetime
import time
import sys

EPSILON = np.finfo(np.double).eps
FRAME_TIME_LENGTH = 50  # length of frame in milliseconds
#DIVISIONS = np.array([40, 70, 110, 150, 200, 250, 300, 400, 500, 750, 1000, 1500, 2000, 3000, 5000, 11025])
DIVISIONS = np.array([500, 1500, 2000, 2500, 3000, 3500, 4000, 5000, 7000, 10000])
LONG_TERM_MOVING_AVERAGE_LENGTH = 2000 / FRAME_TIME_LENGTH  # length in number of FFTs
SHORT_TERM_MOVING_AVERAGE_LENGTH = 500 / FRAME_TIME_LENGTH
NETWORK_LEARNING_RATE = 0.2
NETWORK_MOMENTUM = 0.5
NETWORK_HIDDEN_NEURONS = 20
NETWORK_ITERATIONS = 100


class AudioBuffer:
    def __init__(self, fft_sample_length, overlap_sample_length):
        self.data = []
        self.fft_sample_length = fft_sample_length
        self.overlap_sample_length = overlap_sample_length
        self.step = fft_sample_length - overlap_sample_length

    def push_samples(self, samples):
        """
        Adds samples to end of buffer data.
        :param samples:
        """
        self.data.extend(samples)

    def pop_working_set(self):
        """
        Returns a piece of the data for then performing FFT analysis.
        Keeps the remainder of the data beyond the FFT sample interval.
        :rtype : list
        """
        length = len(self.data)
        if length < self.fft_sample_length:
            return False
        else:
            count = int((length - self.fft_sample_length) / self.step)
            output_length = self.fft_sample_length + count * self.step
            output = self.data[:output_length + 1]
            self.data = self.data[output_length + 1:]
            return output


class DataBuffer:
    def __init__(self, length=1000):
        self.length = length
        self.data = []

    def push(self, item):
        self.data.append(item)
        self._trim()

    def push_multiple(self, items):
        self.data.extend(items)
        self._trim()

    def _trim(self):
        length = len(self.data)
        if length > self.length:
            self.data = self.data[length - self.length:]


class Classifier(object):  # interface for a generic classifier
    def __init__(self, n_inputs, n_outputs):
        pass

    def train(self, data):
        pass

    def run(self, feature_vector):
        pass


class NeuralNetworkClassifier(Classifier):
    def __init__(self, n_inputs, n_outputs, n_hidden=NETWORK_HIDDEN_NEURONS):
        super(NeuralNetworkClassifier, self).__init__(n_inputs, n_outputs)
        self.network = buildNetwork(n_inputs, n_hidden, n_outputs)
        self.dataset = SupervisedDataSet(n_inputs, n_outputs)

    def train(self, data, iterations=NETWORK_ITERATIONS):
        for item in data:
            self.dataset.addSample(item[0], item[1])
        trainer = BackpropTrainer(self.network, self.dataset, learningrate=NETWORK_LEARNING_RATE, momentum=NETWORK_MOMENTUM)
        for i in xrange(iterations):
            error = trainer.train()
            print (i+1), error
        return error

    def run(self, feature_vector):
        return self.network.activate(feature_vector)[0]


class FeatureVectorBuffer(DataBuffer):
    def __init__(self, length=float("inf")):
        DataBuffer.__init__(self, length)
        self.results = DataBuffer(length)

    def add_vector(self, feature_vector):
        DataBuffer.push(self, feature_vector)
        result = self.classify(feature_vector)
        self.results.push(result)

    def classify(self, feature_vector):
        pass


class FeatureVectorExtractor:
    def __init__(self, rate):
        self.rate = rate
        frame_samples_length = int(float(FRAME_TIME_LENGTH) / float(1000) * float(self.rate))
        self.fft_sample_length = int(2 ** self.nextpow2(frame_samples_length))
        self.overlap_sample_length = int(0.3 * frame_samples_length)
        self.audio_buffer = AudioBuffer(fft_sample_length=self.fft_sample_length,
                                        overlap_sample_length=self.overlap_sample_length)
        # self.buffers = {
        #     "raw_slices": DataBuffer(),
        #     "slices": DataBuffer(),
        #     "zero_crossing_rates": DataBuffer(),
        #     "rolloff_freqs": DataBuffer(),
        #     "slices_bins": DataBuffer()
        # }
        self.buffers = {name: DataBuffer() for name in
                        ["raw_slices", "slices", "zero_crossing_rates", "rolloff_freqs", "slices_bins", "third_octave", "third_octave_autocorrelation"]}

        self.classifier = FeatureVectorBuffer()

    def nextpow2(self, num):
        return int(np.ceil(np.log2(num)))

    def plot_spectrogram(self, bins, freqs, slices, logscale=True, axes=plt):
        power = slices.T
        if logscale:
            z = np.log10(power)
        else:
            z = power
        axes.pcolormesh(bins, freqs, z)

    def find_indexes(self, freqs, divisions):
        # Determine where the divisions are in the freqs list

        indexes = []
        i = 0
        for div in divisions:
            while i < len(freqs) and freqs[i] < div:
                i += 1
            indexes.append(i)

        return indexes

    def list_sum(self, list_of_matrices):
        total = list_of_matrices[0]
        for i in xrange(1, len(list_of_matrices)):
            total = [sum(pair) for pair in zip(total, list_of_matrices[i])]
        return total

    def freq_bins(self, freqs, slices, divisions):
        # Divide slices into frequency bins, returns new slices

        indexes = self.find_indexes(freqs, divisions)

        power = slices.T
        output = []

        prev_index = 0
        for index in indexes:
            output.append(sum(power[prev_index:index + 1]))
            prev_index = index

        output = np.array(output).T

        return output

    def moving_average(self, number):
        slices = self.buffers["raw_slices"].data
        averages = []
        length = len(slices)
        for end in xrange(length - number, length):
            start = max(0, end - LONG_TERM_MOVING_AVERAGE_LENGTH - 1)
            actual_length = end - start + 1
            average = sum(slices[start:end + 1]) / actual_length
            # note there is some imprecision in using integer instead of float math
            # but this is faster, and easier to implement
            averages.append(average)
        averages = np.array(averages)
        return averages

    def trim_outliers(self, data, num_std_devs=3):
        data10 = np.log10(data)

        sd10 = np.std(data10)
        mean10 = np.average(data10)

        #output = min(data, mean10+num_std_devs*sd10)
        #output = max(output, mean10-num_std_devs*sd10)

        output = np.copy(data)

        lower_bound10 = mean10 - num_std_devs * sd10
        upper_bound10 = mean10 + num_std_devs * sd10

        lower_bound = 10 ** lower_bound10
        upper_bound = 10 ** upper_bound10

        # print upper_bound10, lower_bound10

        num_high = 0
        num_low = 0
        count = 0

        for elem in np.nditer(output, op_flags=['readwrite']):
            elem10 = np.log10(elem)
            if elem10 > upper_bound10:
                elem[...] = upper_bound
                num_high += 1
            elif elem10 < lower_bound10:
                elem[...] = lower_bound
                num_low += 1
            count += 1

        # print "Trimmed", num_high, "high,",  num_low, "low samples out of", count
        return output

    def slice_rolloff_freq(self, slice, threshold=0.90):
        target = threshold * sum(slice)
        partial = 0.0
        i = 0
        length = len(slice)
        while partial < target and i < length - 1:
            partial += slice[i]
            i += 1
        return i

    def all_rolloff_freq(self, freqs, slices):
        return [freqs[self.slice_rolloff_freq(x)] for x in slices]

    def avg_zero_crossing_rate(self, sound_data):
        signs = np.sign(np.array(sound_data))
        total = 0
        for i in xrange(1, len(signs)):
            if signs[i - 1] != signs[i]:
                total += 1
        rate = float(total) / len(sound_data)
        return rate

    def normalize(self, slices):
        output = []
        for slice in slices:
            output.append(slice / np.average(slice))
        return np.array(output)

    def _step_length(self):
        return self.fft_sample_length - self.overlap_sample_length

    def high_pass_filter(self, slices, freqs, cutoff_frequency):
        """
        Zeros the frequencies below the specified frequency
        (or the next lowest present)
        and returns the remaining higher frequencies.
        :param slices:
        """
        # Find the index to cut off at
        index = 0
        length = len(freqs)
        while freqs[index] < cutoff_frequency and index < length - 1:
            index += 1

        # Perform the filtering
        output = []

        for slice in slices:
            new_slice = [EPSILON] * index
            new_slice.extend(list(slice[index:]))
            output.append(new_slice)

        output = np.array(output)
        return output

    def autocorrelation_coefficient(self, series):
        series1 = series - np.average(series)
        series2 = series1[::-1]
        corr = np.correlate(series, series2)
        return float(corr) / np.var(series)

    def analyze(self, data):
        """

        :param data:
        :return:
        """
        (Pxx, freqs, t) = mlab.specgram(x=data, NFFT=self.fft_sample_length, Fs=self.rate,
                                        noverlap=self.overlap_sample_length)

        raw_slices = Pxx.T  # transpose the power matrix into time slices
        n = len(raw_slices)  # number of slices in each of following sequences

        # Add raw slices to buffer for use in calculating moving average
        self.buffers["raw_slices"].push_multiple(raw_slices)

        # Normalize the slices for analysis purposes
        slices = abs(raw_slices - self.moving_average(len(raw_slices)))  # subtract baseline long-term moving average
        slices[slices == 0] = EPSILON  # replace zero values with small number to prevent invalid logarithm
        slices = self.trim_outliers(slices)  # trim outliers from data
        self.buffers["slices"].push_multiple(slices)

        # Calculate zero-crossing rates (in intervals of the FFT block size, w/ overlap)
        zero_crossing_rates = []
        for section in self._raw_data_in_slices(data):
            zero_crossing_rates.append(self.avg_zero_crossing_rate(section))
        self.buffers["zero_crossing_rates"].push_multiple(zero_crossing_rates)

        # Calculate rolloff frequencies, with high-pass filter
        filtered_slices = self.high_pass_filter(slices, freqs, 500)
        rolloff_freqs = self.all_rolloff_freq(freqs, filtered_slices)
        self.buffers["rolloff_freqs"].push_multiple(rolloff_freqs)

        # Divide each slice into frequency bins
        slices_bins = self.freq_bins(freqs, slices, DIVISIONS)
        self.buffers["slices_bins"].push_multiple(slices_bins)

        # Extract the third octave
        third_octave_indexes = self.find_indexes(freqs, [700, 1300])
        third_octave = [slice[third_octave_indexes[0]:third_octave_indexes[1]] for slice in slices]
        self.buffers["third_octave"].push_multiple(third_octave)

        # Third octave autocorrelation
        third_octave_autocorrelation = []
        for slice in third_octave:
            third_octave_autocorrelation.append(self.autocorrelation_coefficient(slice))
        self.buffers["third_octave_autocorrelation"].push_multiple(third_octave_autocorrelation)

        # Create feature vectors
        vectors = []
        for i in xrange(n):
            vector = []
            vector.extend(np.log10(slices_bins[i]))
            vector.append(zero_crossing_rates[i])
            vector.append(third_octave_autocorrelation[i])
            vector.append(np.log10(rolloff_freqs[i]))
            vector = np.array(vector)
            vector = vector / 10
            #vector = np.log10(vector)
            vectors.append(vector)

        # Return vectors
        return vectors

    def _raw_data_in_slices(self, data):
        num = int((len(data) - self.fft_sample_length) / self._step_length()) + 1
        prev_index = 0
        for i in xrange(num):
            section = data[prev_index:prev_index + self.fft_sample_length]
            prev_index += self._step_length()
            yield section

    def push(self, samples):
        self.audio_buffer.push_samples(samples)
        data = self.audio_buffer.pop_working_set()
        if data:
            return self.analyze(data)

    def display(self):
        fig, axes = plt.subplots(len(self.buffers) + 1)
        i = 0
        for name in self.buffers.keys():
            print name
            axis = axes[i]
            self._display_buffer(self.buffers[name], axis)
            i += 1
        # self._display_buffer(self.classifier, axes[-1])  # Display feature vector
        plt.show()

    def _display_buffer(self, buffer, axis):
        buffer_data = buffer.data
        if type(buffer_data[0]) is np.ndarray:
            # print as spectrogram
            self.plot_spectrogram(np.array(range(len(buffer_data))), np.array(range(len(buffer_data[0]))),
                                  np.array(buffer_data), axes=axis)
        else:
            # plot as standard (x,y)
            axis.plot(range(len(buffer_data)), buffer_data)

    def process_vector(self, vector):
        self.classifier.add_vector(vector)


class RealtimeAnalyzer:
    def __init__(self, rate, classifier):
        """

        :param rate: int
        :param classifier: Classifier
        """
        self.classifier = classifier
        self.extractor = FeatureVectorExtractor(rate)

    def push(self, samples):
        feature_vectors = self.extractor.push(samples)
        if feature_vectors is not None:
            for vector in feature_vectors:
                result = self.classifier.run(vector)
                self._output(result)

    def _output(self, result):
        if not math.isnan(result):
            scale = 20
            value = min(max(int(result * scale), 0), scale)
            # sys.stdout.flush()
            print "[{0}{1}] {2}".format('#'*value, ' '*(scale-value), result)


VIRTUAL_BUFFER_SIZE = 1000


class FileProcessor(object):
    def _process_file(self, filename):
        rate, data = wavfile.read(filename)
        extractor = FeatureVectorExtractor(rate)
        feature_vectors = extractor.push(data)
        #extractor.display()
        return feature_vectors


class BatchFileTrainer(FileProcessor):
    def __init__(self, classifier):
        self.classifier = classifier
        self.data = []

    def add(self, filename, results):
        feature_vectors = self._process_file(filename)

        # Remove the first vector, which usually has problems
        feature_vectors = feature_vectors[1:]

        # Create a stretched results list the same length as feature_vectors
        target_length = len(feature_vectors)
        source_length = len(results)
        results_stretched = []
        for i in xrange(target_length):
            results_stretched.append(results[int(float(i) / target_length * source_length)])

        data = []
        for i in xrange(target_length):
            item = [feature_vectors[i], results_stretched[i]]
            print item
            data.append(item)

        self.data.extend(data)

    def train(self):
        # Create a new classifier if necessary
        if inspect.isclass(self.classifier):
            self.classifier = self.classifier(len(self.data[0][0]), len(self.data[0][1]))
        return self.classifier.train(self.data)


class FileAnalyzer(FileProcessor):
    def __init__(self, classifier):
        self.classifier = classifier

    def analyze(self, filename):
        vectors = self._process_file(filename)
        results = []
        for vector in vectors:
            result = self.classifier.run(vector)
            print vector
            print result
            results.append(result)
        return results