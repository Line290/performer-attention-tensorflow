import tensorflow as tf
import math
from functools import partial
import numpy as np

def get_shape_list(tensor, expected_rank=None, name=None):
    """Returns a list of the shape of tensor, preferring static dimensions.
    Args:
      tensor: A tf.Tensor object to find the shape of.
      expected_rank: (optional) int. The expected rank of `tensor`. If this is
        specified and the `tensor` has a different rank, and exception will be
        thrown.
      name: Optional name of the tensor for the error message.
    Returns:
      A list of dimensions of the shape of tensor. All static dimensions will
      be returned as python integers, and dynamic dimensions will be returned
      as tf.Tensor scalars.
    """
    if name is None:
        name = tensor.name

    if expected_rank is not None:
        assert_rank(tensor, expected_rank, name)

    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape

def shape_list(x):
    """
    Deal with dynamic shape in tensorflow cleanly.

    Args:
        x (:obj:`tf.Tensor`): The tensor we want the shape of.

    Returns:
        :obj:`List[int]`: The shape of the tensor as a list.
    """
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]


def nonnegative_softmax_kernel_feature_creator(data,
                                               projection_matrix,
                                               is_query,
                                               normalize_data=True,
                                               eps=0.000001):
    """Constructs nonnegative kernel features for fast softmax attention.
    Args:
    data: input for which features are computes
    projection_matrix: random matrix used to compute features
    attention_dims_t: tuple of attention dimensions
    batch_dims_t: tuple of batch dimensions
    precision: precision parameter
    is_query: predicate indicating whether input data corresponds to queries or
      keys
    normalize_data: predicate indicating whether data should be normalized,
    eps: numerical stabilizer.
    Returns:
    Random features for fast softmax attention.
    """
    if data.dtype != projection_matrix.dtype:
        projection_matrix = tf.saturate_cast(projection_matrix, data.dtype)

    if normalize_data:
        # We have e^{qk^T/sqrt{d}} = e^{q_norm k_norm^T}, where
        # w_norm = w * data_normalizer for w in {q,k}.
        data_shape = get_shape_list(data)
        data_normalizer = 1.0 / (math.sqrt(math.sqrt(float(data_shape[-1]))))
    else:
        data_normalizer = 1.0
    ratio = 1.0 / math.sqrt(float(get_shape_list(projection_matrix)[0]))
    # data_mod_shape = data.shape[:len(data.shape)-2] + projection_matrix.shape
    data_mod_shape = get_shape_list(data)[:len(data.shape)-2] + get_shape_list(projection_matrix)
    data_thick_random_matrix = tf.zeros(data_mod_shape, dtype=data.dtype) + projection_matrix  # broadcast to batch axis

    data_dash = tf.einsum('...id,...jd->...ij', (data_normalizer*data), data_thick_random_matrix)

    diag_data = data**2
    diag_data = tf.reduce_sum(diag_data, axis=-1)
    diag_data = (diag_data / 2.0) * data_normalizer**2
    diag_data = tf.expand_dims(diag_data, axis=-1)
    
    if is_query:
        data_dash = ratio * (
            tf.exp(data_dash - diag_data - tf.reduce_max(data_dash, axis=-1, keep_dims=True)) + eps)
    else:
        data_dash = ratio * (
            tf.exp(data_dash - diag_data - tf.reduce_max(data_dash)) + eps)
    
    return data_dash

@tf.custom_gradient
def my_eig(x):
    e, v = np.linalg.qr(x)
    def grad(grad_e, grad_v):
        return None
    return (e, v), grad

@tf.custom_gradient
def qr_wo_grad(x):
    q, r = tf.qr(x, full_matrices=False)
    q, r = tf.stop_gradient(q), tf.stop_gradient(r)
    def grad(dq, dr):
        return dq
    return (q, r), grad

def orthogonal_matrix_chunk(cols, dtype):
    use_numpy = False
    if use_numpy:
        unstructured_block = tf.random_normal((cols, cols), dtype=tf.float32)
        # with tf.GradientTape() as tape:
        #     tape.watch(unstructured_block)
        q, _ = tf.py_function(func=my_eig, inp=[unstructured_block], Tout=[tf.float32, tf.float32])
        q.set_shape(unstructured_block.get_shape())
        q = tf.saturate_cast(q, dtype=dtype)
        # print(q.shape)
    else:
        # unstructured_block = tf.stop_gradient(tf.random_normal((cols, cols), dtype=dtype))
        # q, r = tf.qr(unstructured_block, full_matrices=False)
        # q, r = tf.stop_gradient(q), tf.stop_gradient(r)
        # q, r = qr_wo_grad(unstructured_block)
        unstructured_block = tf.random_normal((cols, cols), dtype=tf.float32)
        q, r = tf.qr(unstructured_block, full_matrices=False)
    return tf.transpose(q)


def gaussian_orthogonal_random_matrix(nb_rows, nb_columns, scaling = 0, dtype=tf.float16):
    nb_full_blocks = int(nb_rows / nb_columns)

    block_list = []

    for _ in range(nb_full_blocks):
        q = orthogonal_matrix_chunk(nb_columns, dtype=dtype)
        block_list.append(q)

    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0:
        q = orthogonal_matrix_chunk(nb_columns, dtype=dtype)
        block_list.append(q[:remaining_rows])

    final_matrix = tf.saturate_cast(tf.concat(block_list, 0), dtype=dtype)

    if scaling == 0:
        multiplier = tf.norm(tf.random_normal((nb_rows, nb_columns), dtype=dtype), axis=1)
    elif scaling == 1:
        multiplier = math.sqrt((float(nb_columns))) * tf.ones((nb_rows,), dtype=dtype)
    else:
        raise ValueError(f'Invalid scaling {scaling}')

    return tf.matmul(tf.diag(multiplier), final_matrix)


def np_orthogonal_matrix_chunk(cols):
    unstructured_block = np.random.normal(size=(cols, cols))
    q, _ = np.linalg.qr(unstructured_block)
    return q.T


def np_gaussian_orthogonal_random_matrix(nb_rows, nb_columns, scaling = 0, dtype=tf.float16):
    nb_full_blocks = int(nb_rows / nb_columns)

    block_list = []

    for _ in range(nb_full_blocks):
        q = np_orthogonal_matrix_chunk(nb_columns)
        block_list.append(q)

    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0:
        q = np_orthogonal_matrix_chunk(nb_columns)
        block_list.append(q[:remaining_rows])

    final_matrix = np.concatenate(block_list, axis=0)
    final_matrix = tf.convert_to_tensor(final_matrix, dtype=dtype)
    if scaling == 0:
        multiplier = tf.norm(tf.random_normal((nb_rows, nb_columns), dtype=dtype), axis=1)
    elif scaling == 1:
        multiplier = math.sqrt((float(nb_columns))) * tf.ones((nb_rows,), dtype=dtype)
    else:
        raise ValueError(f'Invalid scaling {scaling}')

    return tf.matmul(tf.diag(multiplier), final_matrix)


# for bidirectional/masked language modelling
def linear_attention(q, k, v):
    context = tf.einsum('...nd,...ne->...de', k, v)
    out = tf.einsum('...de,...nd->...ne', context, q)
    return out

# for unidirectional/causal modelling
# def causal_linear_attention(q, k, v):
#     k_cumsum = tf.cumsum(k, axis=-2)
#     context = tf.einsum('...nd,...ne->...nde', k, v)
#     context = tf.cumsum(context, axis=-3)
#     context /= tf.expand_dims(k_cumsum, axis=-1)
#     out = tf.einsum('...nde,...nd->...ne', context, q)
#     return out


def causal_linear_attention(qs, ks, vs):  # [bs, num_heads, len, head_dims]

    qs = tf.transpose(qs, (2, 0, 1, 3))
    ks = tf.transpose(ks, (2, 0, 1, 3))
    vs = tf.transpose(vs, (2, 0, 1, 3))
    # z_slice_shape = (ks.shape[1], ks.shape[2], ks.shape[-1], vs.shape[-1])
    ks_shape = shape_list(ks)
    vs_shape = shape_list(vs)
    z_slice_shape = ks_shape[1:] + vs_shape[-1:]
    def body(p, qkv):
        (q, k, v) = qkv
        tmp = tf.einsum('...m,...d->...md', k, v)
        tmp_p = p[0] + tmp
        X_slice = tf.einsum('...m,...md->...d', q, tmp_p)
        return tmp_p, X_slice

    init_value = (tf.zeros(z_slice_shape, dtype=qs.dtype),
                  tf.zeros(vs_shape[1:], dtype=qs.dtype))
    p, W = tf.scan(body, (qs, ks, vs), init_value)
    return tf.transpose(W, (1, 2, 0, 3))  # [bs, num_heads, len, head_dims]

def _denominator(qs, ks):
    # [bs, num_heads, len, head_dims] -> [len, bs, num_heads, head_dim]
    qs = tf.transpose(qs, (2, 0, 1, 3))
    ks = tf.transpose(ks, (2, 0, 1, 3))
    qs_shape = shape_list(qs)
    t_slice_shape = qs_shape[1:]    # (bs, num_heads, head_dim)
    res_shape = qs_shape[1:-1]
    def body(p, qk):
        q, k = qk
        tmp = p[0] + k
        x = tf.einsum('...m,...m->...', q, tmp)
        return tmp, x

    init_value = (tf.zeros(t_slice_shape, dtype=qs.dtype),
                  tf.zeros(res_shape, dtype=qs.dtype))
    p, R = tf.scan(body, (qs, ks), init_value) # R: (len, bs, num_heads)
    return tf.transpose(R, (1,2,0))



def fast_attention(q, k, v,
                   dim_heads,
                   nb_features=256,
                   redraw_projection=True,
                   ortho_scaling=0,
                   lm_type='bi',  # unibi, bi, plm
                   out_proj_mat=False,
                   renormalize_attention=True,
                   numerical_stabilizer=1e-6):
    '''
    :param q: shape # [batch_size, num_heads, len, head_dims]
    :param k: same shape with q
    :param v: same shape with q
    :param dim_heads: head_dims
    :param nb_features: dimension of projection matrix
    :param redraw_projection: use random projection matrix in each mini-batch
    :param ortho_scaling:
    :param lm_type: type of attention
    :param out_proj_mat: is or not output projection matrix
    :param renormalize_attention: (very important)
    :param numerical_stabilizer:
    :return:
    '''
    # q = tf.saturate_cast(q, tf.float32)
    # k = tf.saturate_cast(k, tf.float32)
    # v = tf.saturate_cast(v, tf.float32)
    if redraw_projection:
        # random gaussian orthogonal random matrix for every training iteration
        projection_matrix = gaussian_orthogonal_random_matrix(nb_rows=nb_features,
                                                              nb_columns=dim_heads,
                                                              scaling=ortho_scaling,
                                                              dtype=q.dtype)
        # print("redraw")
    else:
        # fixed gaussian orthogonal random matrix for every training iteration
        projection_matrix = np_gaussian_orthogonal_random_matrix(nb_rows=nb_features,
                                                                 nb_columns=dim_heads,
                                                                 scaling=ortho_scaling,
                                                                 dtype=q.dtype)
        # print("no-redraw")

    create_kernel = partial(nonnegative_softmax_kernel_feature_creator,
                            projection_matrix=projection_matrix, eps=numerical_stabilizer)
    q_prime = create_kernel(q, is_query=True) # [bs, num_heads, len, head_dims]
    k_prime = create_kernel(k, is_query=False)

    if lm_type == 'bi':
        out = linear_attention(q_prime, k_prime, v)
        if not renormalize_attention:
            if out_proj_mat:
                return (out, projection_matrix)
            else:
                return out
        else:
            # Construct T = (K^{'})^{T} 1_L
            T = tf.reduce_sum(k_prime, axis=2,
                              keep_dims=False)  # [bs, num_heads, len, head_dims] -> [bs, num_heads, head_dims]
            # Construct partition function: R = Q^{'} T = Q^{'}(K^{'})^{T} 1_L
            R = tf.einsum('...nd,...d->...n', q_prime, T)
    elif lm_type == 'unibi':
        out = causal_linear_attention(q_prime, k_prime, v)
        if not renormalize_attention:
            if out_proj_mat:
                return (out, projection_matrix)
            else:
                return out
        else:
            R = _denominator(q_prime, k_prime)

    elif lm_type == 'plm':
        NotImplementedError("Need to implement")
    R_shape = shape_list(R)
    R_zero_mask = tf.zeros(R_shape, dtype=R.dtype)
    R_numerical_stabilizer_mask = R_zero_mask + 2*numerical_stabilizer
    # R_add_numerical_stabilizer = tf.where(tf.abs(R) <= numerical_stabilizer, 2*numerical_stabilizer, 0.)
    R_add_numerical_stabilizer = tf.where(tf.abs(R) <= numerical_stabilizer, R_numerical_stabilizer_mask, R_zero_mask)
    R = R + R_add_numerical_stabilizer
    R = tf.expand_dims(tf.reciprocal(R), axis=-1) # [bs, num_heads, len] -> [bs, num_heads, len, 1]
    out = out * R
    # out = tf.saturate_cast(out, tf.float16)
    # [bs, num_heads, len, head_dims]
    if out_proj_mat:
        return (out, projection_matrix)
    else:
        return out