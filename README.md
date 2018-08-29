# Sitcom Laughtrack mute tool
### or *a LSTM network for audio classification*

Jeff Green, 2018

### Summary

A tool to mute laugh tracks from an audio clip without operator intervention.  Implements a recurrent neural network (LSTM network) to classify audio.

### Example Output

See https://youtu.be/DeTQBiKzmYc for a clip with the laugh track muted using this tool.

### Description

I made this specifically for muting the audience laugh track from the TV show Frasier.

Because the show's "laugh track" is not pre-canned laugher, but instead recorded from the live studio, it varies significantly between instances.  As a result, while the "laugh" audio can certainly be classified, it requires more than a simple matching algorithm to classify correctly.

I use the `librosa` library for feature extraction from the audio, create samples using a rolling window, and then apply a 3-layer LSTM network to classify each sample.

Technically, this could be used as a generic two-class audio classifier, but the model is not validated against anything besides laugh audio classification.

### Index

`./assets/` contains a model trained on my labelled examples.  I used this model to create the example output video above.

`./src/` contains the tool, which can be run at the command line.  I also added a helper script for Windows that wraps the necessary ffmpeg calls for splitting and combining video and audio tracks.

`./src/benchmark.py` is a mess of code that I used to train and evaluate competing model variants.  It may or may not work with the released `laughr.py` version.

The `mutelaugh.ipynb` Jupyter notebook contains step-by-step explanations and visualizations to help understand how the tool works.

### Installing and Using

To install, use `pipenv`:
```
git clone https://github.com/jeffgreenca/laughr.git
cd laughr
cd src
pipenv install
```

Run the `laughr.py` script with --help for usage information.
```
pipenv run python laughr.py --help
```

Output:
```
usage: laughr.py [-h] --model MODEL.h5
                 [--train /path/to/L*.wav /path/to/D*.wav]
                 [--mute-laughs SOURCE.wav OUTPUT.wav]

A tool to mute laugh tracks from an audio clip automatically. For example, to
remove laugh tracks from audio clips of 90's TV sitcoms. Implements a
recurrent neural network (LSTM network) to classify audio, then transforms
(mutes) the "laugh" class. (jeffgreenca, 2018)

optional arguments:
  -h, --help            show this help message and exit

Commands:
  You can train the model, mute laughs, or do both in one command.
  Alternatively, specify only --model to print a summary.

  --model MODEL.h5      When training, the Keras model is saved to this file
                        (overwrites!). When running only --mute-laughs, the
                        model is loaded from this file.
  --train /path/to/L*.wav /path/to/D*.wav
                        Train the model on the examples. Each glob should
                        specify a set of audio files containing laugher, and a
                        set containing non-laugher, respectively. You might
                        use a tool like Audacity to label and "Export
                        Multiple" to speed up creation of the training set.
  --mute-laughs SOURCE.wav OUTPUT.wav
                        Identifies laugher in the source file, mutes it, and
                        saves the result in the output file.

```

# License

MIT license applies, except for `benchmark.py` which is released under CRAPL.
