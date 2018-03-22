#!/usr/bin/python3
import numpy as np
import soundfile as sf
import librosa
import math
from glob import glob
import argparse
import os


class RawClip3(object):
    featureFuncs = ['tonnetz', 'spectral_rolloff', 'spectral_contrast',
                    'spectral_bandwidth', 'spectral_flatness', 'mfcc',
                    'chroma_cqt', 'chroma_cens', 'melspectrogram']

    def __init__(self, sourcefile, Y_class=None):
        self.y, self.sr = sf.read(sourcefile)
        self.laughs = None
        self.Y_class = Y_class

    def resample(self, rate, channel):
        return librosa.resample(self.y.T[channel], self.sr, rate)

    def amp(self, rate=22050, n_fft=2048, channel=0):
        D = librosa.amplitude_to_db(librosa.magphase(librosa.stft(
            self.resample(rate, channel), n_fft=n_fft))[0], ref=np.max)
        return D

    def _extract_feature(self, func):
        method = getattr(librosa.feature, func)

        # Construct params for each 'class' of features
        params = {'y': self.raw}
        if 'mfcc' in func:
            params['sr'] = self.sr
            params['n_mfcc'] = 128
        if 'chroma' in func:
            params['sr'] = self.sr

        feature = method(**params)

        return feature

    def _split_features_into_windows(self, data, duration):
        # Apply a moving window
        windows = []

        # Pad the rightmost edge by repeating frames, simplifies stretching
        # the model predictions to the original audio later on.
        data = np.pad(data, [[0, duration], [0, 0]], mode='edge')
        for i in range(data.shape[0] - duration):
            windows.append(data[i:i + duration])

        return np.array(windows)

    def build_features(self, duration=30, milSamplesPerChunk=10):
        # Extract features, one chunk at a time (to reduce memory required)
        # Tip: about 65 million samples for a normal-length episode
        # 10 million samples results in around 1.5GB to 2GB memory use
        features = []

        chunkLen = milSamplesPerChunk * 1000000
        numChunks = math.ceil(self.y.shape[0] / chunkLen)

        for i in range(numChunks):
            # Set raw to the current chunk, for _extract_feature
            self.raw = self.y.T[0][i * chunkLen:(i + 1) * chunkLen]

            # For this chunk, run all of our feature extraction functions
            # Each returned array is in the shape (features, steps)
            # Use concatenate to combine (allfeatures, steps)
            chunkFeatures = np.concatenate(
                list(
                    map(self._extract_feature, self.featureFuncs)
                )
            )
            features.append(chunkFeatures)

        # Transform to be consistent with our LSTM expected input
        features = np.concatenate(features, axis=1).T
        # Combine our chunks along the time-step axis.
        features = self._split_features_into_windows(features, duration)

        return features


class DataSet(object):
    def __init__(self, datapath, laughPrefix='/ff*.wav', dialogPrefix='/dd*.wav'):
        self.clips = []
        for y_class, files in [[1., 0.], glob(datapath + laughPrefix)], [[0., 1.], glob(datapath + dialogPrefix)]:
            for ff in files:
                self.clips.append(RawClip3(ff, y_class))
        np.random.seed(seed=0)
        self.X, self.Y_class = self._get_samples()
        self.idx_train, self.idx_cv, self.idx_test = self.split_examples_index(
            len(self.Y_class))

    def split_examples_index(self, total):
        """Returns shuffled index for 60/20/20 split of train, cv, test"""
        np.random.seed(seed=0)
        idx = np.random.choice(total, size=total, replace=False, )

        # 60/20/20 split
        train = idx[0:int(total * 0.6)]
        cv = idx[int(total * 0.6):int(total * 0.6) + int(total * 0.2)]
        test = idx[int(total * 0.8):]

        return train, cv, test

    def _get_samples(self):
        X = []
        y = []
        for clip in self.clips:
            for s in clip.build_features():
                X.append(s)
                y.append(clip.Y_class)

        return np.array(X), np.array(y)


class LaughRemover(object):
    def __init__(self, kerasModel=None, kerasModelFile=None):
        import keras
        assert kerasModel or kerasModelFile
        if kerasModel:
            self.model = kerasModel
        elif kerasModelFile:
            self.model = keras.models.load_model(filepath=kerasModelFile)

    def remove_laughs(self, infile, outfile):
        rc = RawClip3(infile)
        rc.laughs = self.model.predict(rc.build_features())
        self._apply_laughs_array(rc.y, rc.sr, outfile, rc.laughs[:, 1])
        return rc

    def _apply_laughs_array(self, y, sr, outfile, laughs):
        y.T[0] = self._apply_frames_to_samples(frames=laughs, samples=y.T[0])
        y.T[1] = self._apply_frames_to_samples(frames=laughs, samples=y.T[1])
        sf.write(outfile, y, sr)

    def _apply_frames_to_samples(self, frames, samples, exp=1, period=15):
        # Apply a rolling average to smooth the laugh/notlaugh sections
        frames = np.convolve(frames, np.ones((period,)) / period, mode='same')
        # Each frame = default 512 samples, so expand over that period
        frames = np.repeat(frames, librosa.core.frames_to_samples(1))
        # Trim excess padding off the rightmost end
        frames = frames[:len(samples)]
        # Finally, apply audio volume change
        return samples * (frames ** exp)


def do_train(nonLaughFiles, laughFiles, modelOutFilename):
    from keras.models import Sequential
    from keras.layers import Dense, LSTM, Dropout, BatchNormalization
    ds = DataSet('', laughPrefix=laughFiles, dialogPrefix=nonLaughFiles)

    input_shape = ds.X[0].shape

    model = Sequential()
    model.add(BatchNormalization(input_shape=input_shape))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(64))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(ds.X[ds.idx_train], ds.Y_class[ds.idx_train],
              epochs=4, batch_size=16, verbose=2)

    model.evaluate(ds.X[ds.idx_cv], ds.Y_class[ds.idx_cv])
    model.evaluate(ds.X[ds.idx_test], ds.Y_class[ds.idx_test])

    model.save(modelOutFilename)
    return model


def do_mute_laughs(sourceFile, outFile, model):
    params = {}
    if type(model) == str:
        params['kerasModelFile'] = model
    else:
        params['kerasModel'] = model

    laughr = LaughRemover(**params)

    laughr.remove_laughs(sourceFile, outFile)


def print_model_summary(model):
    if not os.path.isfile(model):
        print("ERROR: %s doesn't exist" % model)
        exit(1)

    import keras
    model = keras.models.load_model(model)
    print("\n# Model Summary\n")
    print(model.summary())


if __name__ == '__main__':
    # For clarity, define our rather wordy documentation here
    prog_desc = """
    A tool to mute laugh tracks from an audio clip automatically.
    For example, to remove laugh tracks from audio clips of 90's TV sitcoms.
    Implements a recurrent neural network (LSTM network) to classify audio,
    then transforms (mutes) the "laugh" class. (jeffgreenca, 2018)
    """
    cmd_help = """
    You can train the model, mute laughs, or do both in one command.
    Alternatively, specify only --model to print a summary.
    """
    model_help = """
    When training, the Keras model is saved to this file (overwrites!).
    When running only --mute-laughs, the model is loaded from this file.
    """
    train_help = """
    Train the model on the examples. Each glob should
    specify a set of audio files containing laugher, and a
    set containing non-laugher, respectively. You might
    use a tool like Audacity to label and "Export Multiple"
    to speed up creation of the training set.
    """
    mute_help = """
    Identifies laugher in the source file, mutes it, and saves the
    result in the output file.
    """

    # Build the user interface
    parser = argparse.ArgumentParser(description=prog_desc)
    group = parser.add_argument_group('Commands', description=cmd_help)

    group.add_argument('--model', required=True, type=str, metavar='MODEL.h5',
                       help=model_help)
    group.add_argument('--train', type=str, nargs=2,
                       metavar=('/path/to/L*.wav', '/path/to/D*.wav'),
                       help=train_help)
    group.add_argument('--mute-laughs', type=str, nargs=2,
                       metavar=('SOURCE.wav', 'OUTPUT.wav'),
                       help=mute_help)

    args = parser.parse_args()

    if not args.train and not args.mute_laughs:
        print_model_summary(args.model)
        exit(0)

    if args.train:
        localModel = do_train(laughFiles=args.train[0],
                              nonLaughFiles=args.train[1],
                              modelOutFilename=args.model)

    if args.mute_laughs:
        # The user can choose to train and mute, or just to mute.
        # In either case we need a valid model loaded.
        # Specify the model from disk, unless we already loaded it by training.
        if 'localModel' not in vars():
            localModel = args.model
        do_mute_laughs(sourceFile=args.mute_laughs[0],
                       outFile=args.mute_laughs[1],
                       model=localModel)
