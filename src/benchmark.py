# Released under the CRAPL license :)
# See http://matt.might.net/articles/crapl/
import logging
import numpy as np
import time
import laughr
from glob import glob
import keras_metrics
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, SimpleRNN, GRU

def generate_models(input_shape):
    laugh_precision = keras_metrics.precision(label=0)
    laugh_recall    = keras_metrics.recall(label=0)
    metrics = ['accuracy', laugh_precision, laugh_recall]

    models = {}

    rdrop = 0.2
    for i1, i2, i3 in [(24, 16, 8), (8, 8, 4)]:
        name = "GRU3_%s_%s_%s-rdrop-%s" % (i1, i2, i3, rdrop)
        model = Sequential()
        model.add(GRU(i1, recurrent_dropout=rdrop, input_shape=input_shape, return_sequences=True))
        model.add(GRU(i2, recurrent_dropout=rdrop, return_sequences=True))
        model.add(GRU(i3, recurrent_dropout=rdrop))
        model.add(Dense(2, activation='softmax'))
        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy', metrics=metrics)
        models[name] = model

        name = "SRNN_%s_%s_%s-rdrop-%s" % (i1, i2, i3, rdrop)
        model = Sequential()
        model.add(SimpleRNN(i1, recurrent_dropout=rdrop, input_shape=input_shape, return_sequences=True))
        model.add(SimpleRNN(i2, recurrent_dropout=rdrop, return_sequences=True))
        model.add(SimpleRNN(i3, recurrent_dropout=rdrop))
        model.add(Dense(2, activation='softmax'))
        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy', metrics=metrics)
        models[name] = model
    
        name = "LSTM_%s_%s_%s-rdrop-%s" % (i1, i2, i3, rdrop)
        model = Sequential()
        model.add(LSTM(i1, recurrent_dropout=rdrop, input_shape=input_shape, return_sequences=True))
        model.add(LSTM(i2, recurrent_dropout=rdrop, return_sequences=True))
        model.add(LSTM(i3, recurrent_dropout=rdrop))
        model.add(Dense(2, activation='softmax'))
        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy', metrics=metrics)
        models[name] = model

#   name = "SRNN_24_16_8_drop-0.05"
#   model = Sequential()
#   model.add(SimpleRNN(24, input_shape=input_shape, return_sequences=True))
#   model.add(SimpleRNN(16, return_sequences=True))
#   model.add(SimpleRNN(8))
#   model.add(Dropout(0.05))
#   model.add(Dense(2, activation='softmax'))
#   model.compile(optimizer='rmsprop',
#                 loss='categorical_crossentropy', metrics=metrics)
#   models[name] = model
#
#    name = "SRNN_8_8_8_drop-0.05"
#    model = Sequential()
#    model.add(SimpleRNN(8, input_shape=input_shape, return_sequences=True))
#    model.add(SimpleRNN(8, return_sequences=True))
#    model.add(SimpleRNN(4))
#    model.add(Dropout(0.05))
#    model.add(Dense(2, activation='softmax'))
#    model.compile(optimizer='rmsprop',
#                  loss='categorical_crossentropy', metrics=metrics)
#    models[name] = model
#
#   name = "SRNN_8_4_4_drop-0.05"
#   model = Sequential()
#   model.add(SimpleRNN(8, input_shape=input_shape, return_sequences=True))
#   model.add(SimpleRNN(8, return_sequences=True))
#   model.add(SimpleRNN(4))
#   model.add(Dropout(0.05))
#   model.add(Dense(2, activation='softmax'))
#   model.compile(optimizer='rmsprop',
#                 loss='categorical_crossentropy', metrics=metrics)
#   models[name] = model
#
#    name = "SRNN_24_16_drop-0.05"
#    model = Sequential()
#    model.add(SimpleRNN(24, input_shape=input_shape, return_sequences=True))
#    model.add(SimpleRNN(16))
#    model.add(Dropout(0.05))
#    model.add(Dense(2, activation='softmax'))
#    model.compile(optimizer='rmsprop',
#                  loss='categorical_crossentropy', metrics=metrics)
#    models[name] = model
#   
#   for i1, i2, i3 in [(8, 4, 4)]:
#       for drop in [0.05]:
#           name = "LSTM_%s_%s_%s_drop-%s" % (i1, i2, i3, drop)
#           model = Sequential()
#           model.add(LSTM(i1, input_shape=input_shape, return_sequences=True))
#           model.add(LSTM(i2, return_sequences=True))
#           model.add(LSTM(i3))
#           model.add(Dropout(drop))
#           model.add(Dense(2, activation='softmax'))
#           model.compile(optimizer='rmsprop',
#                         loss='categorical_crossentropy', metrics=metrics)
#           models[name] = model
#
#   for i1, i2, i3 in [(8, 8, 8)]:
#       for drop in [0, 0.05, 0.10]:
#           name = "LSTM_%s_%s_%s_drop-%s" % (i1, i2, i3, drop)
#           model = Sequential()
#           model.add(LSTM(i1, input_shape=input_shape, return_sequences=True))
#           model.add(LSTM(i2, return_sequences=True))
#           model.add(LSTM(i3))
#           model.add(Dropout(drop))
#           model.add(Dense(2, activation='softmax'))
#           model.compile(optimizer='rmsprop',
#                         loss='categorical_crossentropy', metrics=metrics)
#           models[name] = model
#
#   for i1, i2 in [(8, 4), (64, 16)]:
#       for drop in [0, 0.05, 0.10]:
#           name = "LSTM_%s_%s_drop-%s" % (i1, i2, drop)
#           model = Sequential()
#           model.add(LSTM(i1, input_shape=input_shape, return_sequences=True))
#           model.add(LSTM(i2))
#           model.add(Dropout(drop))
#           model.add(Dense(2, activation='softmax'))
#           model.compile(optimizer='rmsprop',
#                         loss='categorical_crossentropy', metrics=metrics)
#           models[name] = model

    return models

def do_train(ds, model):
    s = time.time()
    model.fit(ds.X, ds.Y_class, epochs=15, batch_size=1000, verbose=0)
    duration = time.time() - s
    return model, round(duration, 2)

def do_eval(ds, model):
    results = model.evaluate(ds.X, ds.Y_class, verbose=0)
    return dict(zip(model.metrics_names, [round(r, 6) for r in results]))

def split_examples_index(total, seed=0):
    """Returns shuffled index for 60/20/20 split of train, cv, test"""
    np.random.seed()
    idx = np.random.choice(total, size=total, replace=False, )

    train = idx[0:int(total * 0.70)]
    test = idx[int(total * 0.70):]

    return list(train), list(test)

if __name__ == '__main__':

    for windowSize in [40, 20, 80, 30]:
        print("TESTING WITH WNIDOW SIZE ------------------------ windowSize=%s" % windowSize)
        seed=0
        print("USING SEED=%s" % seed)
        
        laughs = glob("training/combined/f*.wav")
        dialogs = glob("training/combined/d*.wav")
        print("using %s laughs and %s nonlaughs" % (len(laughs), len(dialogs)))
        
        laughs_train_idx, laughs_test_idx = split_examples_index(len(laughs), seed=seed)
        dialogs_train_idx, dialogs_test_idx = split_examples_index(len(dialogs), seed=seed)
        print("(laughs train, test, nonlaughs train, test) = (%s)" % [len(x) for x in [laughs_train_idx, laughs_test_idx, dialogs_train_idx, dialogs_test_idx]])
        
        print("Loading datasets")
        ds = []
        ds.append(laughr.DataSet('', laughList=[laughs[i] for i in laughs_train_idx], dialogList=[dialogs[i] for i in dialogs_train_idx], windowDuration=windowSize))
        ds.append(laughr.DataSet('', laughList=[laughs[i] for i in laughs_test_idx], dialogList=[dialogs[i] for i in dialogs_test_idx], windowDuration=windowSize))

        print("Building models")
        modsec = time.time()
        models = generate_models(ds[0].X[0].shape)
        print("Done in %s\n" % round(time.time() - modsec, 2))

        print("\t".join([str(x) for x in ["name", "train-secs", "results"]]))
        for name in models:
            m, s = do_train(ds[0], models[name])
            r    = do_eval(ds[1], m)
            print("\t".join([str(x) for x in [name, s, r]]))
