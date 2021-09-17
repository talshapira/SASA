import pandas as pd
import numpy as np
from keras.preprocessing import sequence

def generate_dataset_and_meta(data_labels_agg, data_type='lats_longs', normalize_lats_longs=True, max_len=40, remove_repetition=False, padding='post'):
    
    sources = data_labels_agg['IPsource'].values
    
    if data_type=='lats_longs':
        if normalize_lats_longs:
            data_labels_agg['lat'] = data_labels_agg['lat'].apply(lambda l: [x/90 for x in l])
            data_labels_agg['long'] = data_labels_agg['long'].apply(lambda l: [x/180 for x in l])
        lats = data_labels_agg['lat'].values
        longs = data_labels_agg['long'].values
        data_labels_agg.drop(["lat", "long", "IPsource"], axis=1, inplace=True)
        
        if remove_repetition:
            for i in range(len(lats)):
                lat, long, source = lats[i], longs[i], sources[i]
                new_lat, new_long, new_source = [lat[0]], [long[0]], [source[0]]
                for j in range(1, len(lat)):
                    if lat[j] != lat[-1] or long[j] != long[-1]:
                        new_lat.append(lat[j])
                        new_long.append(long[j])
                        new_source.append(source[j])
                lats[i], longs[i], sources[i] = new_lat, new_long, new_source
        
        lats = sequence.pad_sequences(lats, maxlen=max_len, dtype='float32', padding=padding)
        longs = sequence.pad_sequences(longs, maxlen=max_len, dtype='float32', padding=padding)
        sources = sequence.pad_sequences(sources, maxlen=max_len, value=0, padding=padding)
        
        lats = lats.reshape((lats.shape[0], lats.shape[1], 1))
        longs = longs.reshape((longs.shape[0], longs.shape[1], 1))
        lats_longs = np.concatenate((lats, longs), axis=2)
        
        data_labels_agg["sources"] = sources.tolist()
        data_labels_agg["lats_longs"] = lats_longs.tolist()
        return data_labels_agg
        
    elif data_type=='countries':
        countries = data_labels_agg['geoCC'].values
        data_labels_agg.drop(["IPsource"], axis=1, inplace=True)
        
        if remove_repetition:
            for i in range(len(countries)):
                country, source = countries[i], sources[i]
                new_country, new_source = [country[0]], [source[0]]
                for j in range(1, len(country)):
                    if country[j] != country[-1]:
                        new_country.append(country[j])
                        new_source.append(source[j])
                countries[i], sources[i] = new_country, new_source

        countries = sequence.pad_sequences(countries, maxlen=max_len, value="", dtype='U2', padding=padding)
        sources = sequence.pad_sequences(sources, maxlen=max_len, value=0, padding=padding)
        countries[countries=="na"] = "-"
        data_labels_agg["sources"] = sources.tolist()
        data_labels_agg["countries"] = countries.tolist()
        return data_labels_agg
    else:
        raise Exception


def dataset_country2ind(dataset, max_len, country_idx, unrecognized_country=0):
    dataset_idx = np.zeros([len(dataset), max_len], dtype=np.int32)

    for i, route in enumerate(dataset):
        for t, country in enumerate(route):
            dataset_idx[i, t] = country_idx.get(country, unrecognized_country)
    return dataset_idx


def generate_set_arrays(df_set, max_len, data_type, labels=None, country_idx=None):
    if data_type == "lats_longs":
        x_set = np.stack(df_set["lats_longs"].values, axis=0)
        del df_set["lats_longs"]
    else:
        x_set = df_set["countries"].values
        x_set = dataset_country2ind(x_set, max_len, country_idx)
        del df_set["countries"]
    
    u_set = np.stack(df_set['sources'].values, axis=0)
    del df_set["sources"]
    if labels:
        y_set = df_set[labels].values
        return x_set, u_set, y_set
    return x_set, u_set
