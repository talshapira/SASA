from self_attention_layers import ScaledDotProductAttention, SourceAwareAttention
from keras.layers import Layer, Input, Dense, Activation, Bidirectional, TimeDistributed, Embedding, Reshape, RepeatVector, Permute, multiply

def SDPA(inputs, hidden_size=None):
    if hidden_size==None:
        hidden_size = int(inputs.shape[2])
    q = TimeDistributed(Dense(hidden_size, use_bias=False), name='q')(inputs)
    k = TimeDistributed(Dense(hidden_size, use_bias=False), name='k')(inputs)
    v = TimeDistributed(Dense(hidden_size, use_bias=False), name='v')(inputs)
    
    attention_mul, scores = ScaledDotProductAttention(name="ScaledDotProductAttention", return_attention=True)([q,k,v])
    return attention_mul, scores

def sources_embedding(sources, hidden_size, sources_num, time_steps, learned=True, single_sources_vector=True):
    if learned:
        if single_sources_vector:
            sources = Embedding(sources_num, 1, input_length=time_steps,name='embedded_sources')(sources)
            sources = Activation('sigmoid')(sources)
            sources = Reshape((time_steps,))(sources)
            sources = RepeatVector(hidden_size)(sources)
            sources = Permute((2, 1),name="sources_out")(sources)
        else:
            sources = Embedding(sources_num, hidden_size, input_length=time_steps,name='embedded_sources')(sources)
            sources = Activation('sigmoid', name='sources_out')(sources)
    else:
        sources = RepeatVector(hidden_size, name="sources_out")(sources)
    return sources
    

def SDPA_Q_S(inputs, sources, sources_num, hidden_size=None, learned=True): # SDPAttention with q=S
    if hidden_size==None:
        hidden_size = int(inputs.shape[2])
   
    sources = sources_embedding(sources, hidden_size, sources_num, int(inputs.shape[1]), learned) ## time_steps
    k = TimeDistributed(Dense(hidden_size, use_bias=False), name='k')(inputs)
    v = TimeDistributed(Dense(hidden_size, use_bias=False), name='v')(inputs)
    
    attention_mul, scores = ScaledDotProductAttention(name="ScaledDotProductAttention", return_attention=True)([sources,k,v])
    return attention_mul, scores, sources, scores #Here sources_scores = scores


def SASA(inputs, sources, sources_num, hidden_size=None, learned=True):
    if hidden_size==None:
        hidden_size = int(inputs.shape[2])
   
    s = sources_embedding(sources, hidden_size, sources_num, int(inputs.shape[1]), learned)
    # print(s.shape)
#     T = s.shape.as_list()[1]
#     print(T)
#     s2 = K.batch_dot(s,  s, axes=[2, 2])/T
#     s = multiply([s, s])
#     print(s2.shape)
#     s = sources_embedding(sources, time_steps)
    q = TimeDistributed(Dense(hidden_size, use_bias=False), name='q')(inputs)
    k = TimeDistributed(Dense(hidden_size, use_bias=False), name='k')(inputs)
    v = TimeDistributed(Dense(hidden_size, use_bias=False), name='v')(inputs)
    # print(q.shape, k.shape, v.shape)
    
    attention_mul, scores, sources_scores = SourceAwareAttention(name="ScaledDotProductAttention", return_attention=True)([q,k,v,s])

    # print(attention_mul.shape, scores.shape, sources_scores.shape)
    return attention_mul, scores, s, sources_scores


def SDPA_QS_KS(inputs, sources, sources_num, hidden_size=None, learned=True): # SDPAttention with q=s*q and k=s*k
    if hidden_size==None:
        hidden_size = int(inputs.shape[2])
   
    s = sources_embedding(sources, hidden_size, sources_num, int(inputs.shape[1]), learned)
    q = TimeDistributed(Dense(hidden_size, use_bias=False), name='q')(inputs)
    k = TimeDistributed(Dense(hidden_size, use_bias=False), name='k')(inputs)
    v = TimeDistributed(Dense(hidden_size, use_bias=False), name='v')(inputs)
    
    q = multiply([s, q])
    k = multiply([s, k])
    # print(q.shape, k.shape, v.shape)
    
    attention_mul, scores = ScaledDotProductAttention(name="ScaledDotProductAttention", return_attention=True)([q,k,v])
    return attention_mul, scores, s, scores #sources_scores = scores


def SDPA_Mul_S(inputs, sources, sources_num, hidden_size=None, learned=True): # SDPAttention + attention_mul*s (apply sources after SDPAttention)
    if hidden_size==None:
        hidden_size = int(inputs.shape[2])
   
    s = sources_embedding(sources, hidden_size, sources_num, int(inputs.shape[1]), learned)
    q = TimeDistributed(Dense(hidden_size, use_bias=False), name='q')(inputs)
    k = TimeDistributed(Dense(hidden_size, use_bias=False), name='k')(inputs)
    v = TimeDistributed(Dense(hidden_size, use_bias=False), name='v')(inputs)
    
    # print(q.shape, k.shape, v.shape)
    attention_mul, scores = ScaledDotProductAttention(name="ScaledDotProductAttention", return_attention=True)([q,k,v])
    attention_mul = multiply([attention_mul, s])
    
    return attention_mul, scores, s, scores #Here sources_scores = scores