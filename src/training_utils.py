import numpy as np
from imblearn.keras import balanced_batch_generator

def balanced_generator(features, labels, batch_size, input_shape, use_embedding=False, random_state=None):
    indexes = np.arange(len(features)).reshape((len(features), 1))
    training_generator, steps_per_epoch = balanced_batch_generator(indexes, labels, 
                                                               batch_size=batch_size, random_state=random_state)
    index = 0
    while True:
        index += 1
        if index > steps_per_epoch:
            training_generator, steps_per_epoch = balanced_batch_generator(indexes, labels, 
                                                               batch_size=batch_size, random_state=random_state)
            index = 1

        batch_indexes, batch_labels = next(training_generator)
        if not use_embedding:
            yield features[batch_indexes].reshape((batch_size,input_shape[0],input_shape[1])), batch_labels
        else:
             yield features[batch_indexes].reshape((batch_size,input_shape[0])), batch_labels


def generator(features, labels, batch_size):
    index = 0
    while True:
        index += batch_size
        if index >= len(features):
            batch_features = np.append(features[index-batch_size:len(features)], features[0:index-len(features)], axis=0)
            batch_labels = np.append(labels[index-batch_size:len(features)], labels[0:index-len(features)], axis=0)
            index -= len(features)
            yield batch_features, batch_labels
        else:
            yield features[index-batch_size:index], labels[index-batch_size:index]


def val_generator(features, labels, val_batch_size):
    index = 0
    while True:
        index += val_batch_size
        batch_features, batch_labels = features[index-val_batch_size:index], labels[index-val_batch_size:index]
        if index >= len(features):
            index = 0
        yield batch_features, batch_labels


def balanced_sources_generator(features, sources, labels, batch_size, input_shape, use_embedding=False, random_state=None):
    indexes = np.arange(len(features)).reshape((len(features), 1))
    training_generator, steps_per_epoch = balanced_batch_generator(indexes, labels, 
                                                               batch_size=batch_size, random_state=random_state)
    index = 0
    while True:
        index += 1
        if index > steps_per_epoch:
            training_generator, steps_per_epoch = balanced_batch_generator(indexes, labels, 
                                                               batch_size=batch_size, random_state=random_state)
            index = 1

        batch_indexes, batch_labels = next(training_generator)
        if not use_embedding:
            yield [features[batch_indexes].reshape((batch_size,input_shape[0],input_shape[1])), sources[batch_indexes].reshape((batch_size,input_shape[0]))], batch_labels
        else:
            yield [features[batch_indexes].reshape((batch_size,input_shape[0])), sources[batch_indexes].reshape((batch_size,input_shape[0]))], batch_labels


def sources_generator(features, sources, labels, batch_size):
    index = 0

    while True:
        index += batch_size
        if index >= len(features):
            batch_features = np.append(features[index-batch_size:len(features)], features[0:index-len(features)], axis=0)
            batch_sources = np.append(sources[index-batch_size:len(features)], sources[0:index-len(features)], axis=0)
            batch_labels = np.append(labels[index-batch_size:len(features)], labels[0:index-len(features)], axis=0)
            index -= len(features)
            yield [batch_features, batch_sources], batch_labels
        else:
            yield [features[index-batch_size:index], sources[index-batch_size:index]] , labels[index-batch_size:index]


def val_sources_generator(features, sources, labels, val_batch_size):
    index = 0

    while True:
        index += val_batch_size
        batch_features, batch_labels = features[index-val_batch_size:index], labels[index-val_batch_size:index]
        batch_sources = sources[index-val_batch_size:index]
        if index >= len(features):
            index = 0
        yield [batch_features, batch_sources], batch_labels
