from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")

def get_model_pred_prob(model, x_test, u_test=None, val_batch_size=1024, use_sources=False):
    if use_sources:
        y_test_prob = model.predict([x_test, u_test], batch_size=val_batch_size, verbose=1)
    else:
        y_test_prob = model.predict(x_test, batch_size=val_batch_size, verbose=1)

    y_test_prediction = np.around(y_test_prob)
    y_test_prediction = y_test_prediction.astype('int')
    return y_test_prediction, y_test_prob


def get_source_embeddings(model, compute_squred=False):
    def sigmoid(x):
        return 1/(1+np.exp(-x))

    embedded_sources = model.get_layer('embedded_sources').get_weights()[0]
    # print(embedded_sources.shape)
    e_s = sigmoid(embedded_sources)
    # print(e_s)
    # print(np.mean(e_s,axis=1))
    if compute_squred:
        s_2 = e_s*e_s.T
        print(s_2)
        print(np.mean(s_2,axis=1))
    return [e[0] for e in e_s]


def get_attention_layer_multi_outputs(model, input_samples, layer_name='ScaledDotProductAttention'):
    layer_output = model.get_layer(layer_name).output
    functor = K.function([model.input], layer_output)
    out = functor([input_samples])
    return out


def get_attention_sources_layer_multi_outputs(model, input_samples, layer_name='ScaledDotProductAttention'):
    layer_output = model.get_layer(layer_name).output
    functor = K.function(model.input, layer_output)
    out = functor(input_samples)
    return out


def get_layer_output(model, input_samples, layer_name='ScaledDotProductAttention'):
    layer_output = model.get_layer(layer_name).output
    functor = K.function(model.input, [layer_output])
    out = functor(input_samples)
    return out[0]


def get_attention_sources_vecs(x_set, u_set, model, use_sources, attention_name, return_au_score=False):
    if use_sources:
        test_samp = [x_set, u_set]
        attention_layer_outs = get_attention_sources_layer_multi_outputs(model, test_samp)
        attention_mul_out = attention_layer_outs[0]
        attention_vec_out = attention_layer_outs[1]
        if attention_name == 'SASA':
            attention_sources_scores = attention_layer_outs[2]

        sources_out = get_layer_output(model, test_samp, 'sources_out')

        if return_au_score:
            return attention_vec_out, sources_out, attention_sources_scores
        return attention_vec_out, sources_out
    
    if not use_sources:
        test_samp = x_set

        attention_layer_outs = get_attention_layer_multi_outputs(model, test_samp)
        attention_mul_out = attention_layer_outs[0]
        attention_vec_out = attention_layer_outs[1]
        return attention_vec_out, None
