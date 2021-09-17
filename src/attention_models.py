from keras.models import Model
from keras.layers import Layer, Input, Dense, Activation, Bidirectional, TimeDistributed, Embedding
from keras.layers.recurrent import LSTM
from keras.layers.normalization import BatchNormalization

from self_attention import SDPA, SDPA_Q_S, SASA, SDPA_QS_KS, SDPA_Mul_S

ATTENTION_NAME_TYPE_DICT = {'SDPA': SDPA, 'SDPA_Q_S': SDPA_Q_S, 'SASA': SASA, 'SDPA_QS_KS': SDPA_QS_KS, 'SDPA_Mul_S': SDPA_Mul_S}

def SimpleBlock(inputs, dilation_rate=2):
    out = Convolution1D(nb_filter=100, dilation_rate=dilation_rate, filter_length=3, border_mode='same')(inputs)
    out = BatchNormalization()(out)
    return Activation('relu')(out)

def TCNBlock(inputs): ### Assuming after regular 1DConv
    out = SimpleBlock(inputs, 1)
    out = SimpleBlock(inputs, 2)
    return SimpleBlock(out, 4)


def LSTM_attention_model_with_sources(input_shape, attention, sources_num=5, use_embedding=False, num_categories=None, embedding_vecor_length=32, no_lstm=False, blstm=True):
    max_len = input_shape[0]
    inputs = Input(shape=input_shape)
    if not use_embedding:
        conv1d_out = TimeDistributed(Dense(32))(inputs)
    else:
        embedding = Embedding(num_categories, embedding_vecor_length, input_length=max_len)(inputs)
        conv1d_out = TimeDistributed(Dense(32))(embedding)
    sources = Input(shape=(max_len,)) #,1
#     conv1d_out = Convolution1D(nb_filter=32, filter_length=3, border_mode='same')(inputs)
    conv1d_out = BatchNormalization()(conv1d_out)
    conv1d_out = Activation('relu')(conv1d_out)
    if no_lstm:
        lstm_out = TCNBlock(conv1d_out)
    else:
        if blstm:
            lstm_out = Bidirectional(LSTM(100, return_sequences=True))(conv1d_out)
        else:
            lstm_out = LSTM(100, return_sequences=True)(conv1d_out)
    attention_mul, scores, e_sources, sources_scores = attention(lstm_out, sources, sources_num)
    attention_mul = Flatten()(attention_mul)
    output = Dense(1, activation='sigmoid', name='output')(attention_mul)
    model = Model(inputs=[inputs, sources], outputs=output) 
    return model

def attention_LSTM_model_with_sources(input_shape, attention, sources_num=5, use_embedding=False, num_categories=None, embedding_vecor_length=32, no_lstm=False, blstm=True):
    max_len = input_shape[0]
    inputs = Input(shape=input_shape)
    if not use_embedding:
        conv1d_out = TimeDistributed(Dense(32))(inputs)
    else:
        embedding = Embedding(num_categories, embedding_vecor_length, input_length=max_len)(inputs)
        conv1d_out = TimeDistributed(Dense(32))(embedding)
    sources = Input(shape=(max_len,)) #,1
#     conv1d_out = Convolution1D(nb_filter=32, filter_length=3, border_mode='same')(inputs)
    conv1d_out = BatchNormalization()(conv1d_out)
    conv1d_out = Activation('relu')(conv1d_out)
    attention_mul, scores, e_sources, sources_scores = attention(conv1d_out, sources, sources_num)

    if no_lstm:
        lstm_out = Flatten()(TCNBlock(attention_mul))
    else:
        if blstm:
#             print("attention_mul", attention_mul.shape)
            lstm_out = Bidirectional(LSTM(100, return_sequences=False))(attention_mul)
        else:
            lstm_out = LSTM(100, return_sequences=False)(attention_mul)
    # print("lstm_out", lstm_out.shape)
    output = Dense(1, activation='sigmoid', name='output')(lstm_out)
    # print("output", output.shape)
    model = Model(inputs=[inputs, sources], outputs=output) 
    return model

def LSTM_attention_model(input_shape, use_embedding=False, num_categories=None, embedding_vecor_length=32, no_lstm=False, blstm=True):
    max_len = input_shape[0]
    inputs = Input(shape=input_shape)
    if not use_embedding:
        conv1d_out = TimeDistributed(Dense(32))(inputs)
    else:
        embedding = Embedding(num_categories, embedding_vecor_length, input_length=max_len)(inputs)
        conv1d_out = TimeDistributed(Dense(32))(embedding)
#     conv1d_out = Convolution1D(nb_filter=32, filter_length=3, border_mode='same')(inputs)
    conv1d_out = BatchNormalization()(conv1d_out)
    conv1d_out = Activation('relu')(conv1d_out)
    if no_lstm:
        lstm_out = TCNBlock(conv1d_out)
    else:
        if blstm:
            lstm_out = Bidirectional(LSTM(100, return_sequences=True))(conv1d_out)
        else:
            lstm_out = LSTM(100, return_sequences=True)(conv1d_out)
    attention_mul, scores = SDPA(lstm_out)
    attention_mul = Flatten()(attention_mul)
    output = Dense(1, activation='sigmoid')(attention_mul)
    model = Model(inputs=[inputs], outputs=output)
    return model

def attention_LSTM_model(input_shape, use_embedding=False, num_categories=None, embedding_vecor_length=32, no_lstm=False, blstm=True):
    ##############################ScaledDotProductAttention
    max_len = input_shape[0]
    inputs = Input(shape=input_shape)
    #     conv1d_out = Convolution1D(nb_filter=32, filter_length=3, border_mode='same')(inputs)
    if not use_embedding:
        conv1d_out = TimeDistributed(Dense(32))(inputs)
    else:
        embedding = Embedding(num_categories, embedding_vecor_length, input_length=max_len)(inputs)
        conv1d_out = TimeDistributed(Dense(32))(embedding)
    conv1d_out = BatchNormalization()(conv1d_out)
    conv1d_out = Activation('relu')(conv1d_out)
    attention_mul, scores = SDPA(conv1d_out)
    if no_lstm:
        lstm_out = TCNBlock(attention_mul)
    else:
        if blstm:
            lstm_out = Bidirectional(LSTM(100, return_sequences=False))(attention_mul)
        else:
            lstm_out = LSTM(100, return_sequences=False)(attention_mul)
    output = Dense(1, activation='sigmoid',name="output")(lstm_out)
    model = Model(inputs=[inputs], outputs=output)
    return model


def generate_model(input_shape, attention_name, use_sources, use_embedding=False, num_categories=5, embedding_vecor_length=32, attention_first=True, blstm=True, no_lstm=False):
    attention = ATTENTION_NAME_TYPE_DICT[attention_name]
    if use_sources:
        if not attention_first:
            model = LSTM_attention_model_with_sources(input_shape, attention=attention, use_embedding=use_embedding, 
                                                      num_categories=num_categories, embedding_vecor_length=embedding_vecor_length,
                                                      no_lstm=no_lstm, blstm=blstm)
        else:
            model = attention_LSTM_model_with_sources(input_shape, attention=attention, use_embedding=use_embedding, 
                                                      num_categories=num_categories, embedding_vecor_length=embedding_vecor_length,
                                                      no_lstm=no_lstm, blstm=blstm)
    else:
        if not attention_first:
            model = LSTM_attention_model(input_shape, use_embedding=use_embedding, 
                                         num_categories=num_categories, embedding_vecor_length=embedding_vecor_length,
                                         no_lstm=no_lstm, blstm=blstm)
        else:
            model = attention_LSTM_model(input_shape, use_embedding=use_embedding, 
                                         num_categories=num_categories, embedding_vecor_length=embedding_vecor_length,
                                         no_lstm=no_lstm, blstm=blstm)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
