from cs231n.layers import *
from cs231n.fast_layers import *


def affine_relu_forward(x, w, b):
  """
  Convenience layer that perorms an affine transform followed by a ReLU

  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, fc_cache = affine_forward(x, w, b)
  out, relu_cache = relu_forward(a)
  cache = (fc_cache, relu_cache)
  return out, cache


def affine_relu_backward(dout, cache):
  """
  Backward pass for the affine-relu convenience layer
  """
  fc_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dx, dw, db = affine_backward(da, fc_cache)
  return dx, dw, db


def conv_relu_forward(x, w, b, conv_param):
  """
  A convenience layer that performs a convolution followed by a ReLU.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  
  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  out, relu_cache = relu_forward(a)
  cache = (conv_cache, relu_cache)
  return out, cache


def conv_relu_backward(dout, cache):
  """
  Backward pass for the conv-relu convenience layer.
  """
  conv_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
  """
  Convenience layer that performs a convolution, a ReLU, and a pool.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  - pool_param: Parameters for the pooling layer

  Returns a tuple of:
  - out: Output from the pooling layer
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  s, relu_cache = relu_forward(a)
  out, pool_cache = max_pool_forward_fast(s, pool_param)
  cache = (conv_cache, relu_cache, pool_cache)
  return out, cache


def conv_relu_pool_backward(dout, cache):
  """
  Backward pass for the conv-relu-pool convenience layer
  """
  conv_cache, relu_cache, pool_cache = cache
  ds = max_pool_backward_fast(dout, pool_cache)
  da = relu_backward(ds, relu_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db


def affine_bn_relu_forward(x, w, b, bn, bn_params):
    a, aff_cache = affine_forward(x, w, b)
    b, bn_cache = batchnorm_forward(a, bn[0, :], bn[1, :], bn_params)
    out, relu_cache = relu_forward(b)
    cache = (aff_cache, bn_cache, relu_cache)
    return out, cache

def affine_bn_relu_backward( dout, cache):
    aff_cache, bn_cache, relu_cache = cache
    db = relu_backward(dout, relu_cache)
    da, dgamma, dbeta = batchnorm_backward(db, bn_cache)
    dx, dw, db = affine_backward(da, aff_cache)
    N = dgamma.shape[0]
    dbn = np.zeros((2, N))
    dbn[0, :] = dgamma
    dbn[1, :] = dbeta
    return dx, dw, db, dbn


def conv_bn_relu_fwd(x, w, b, bn, conv_param, bn_params):
    p, conv_cache = conv_forward_fast(x, w, b, conv_param)
    q, bn_cache = spatial_batchnorm_forward(p, bn[0, :], bn[1, :], bn_params)
    out, relu_cache = relu_forward(q)
    cache = (conv_cache, bn_cache, relu_cache)
    return out, cache

def conv_bn_relu_bkwd(dout, cache):
    conv_cache, bn_cache, relu_cache = cache
    dq = relu_backward(dout, relu_cache)
    dp, dgamma, dbeta = spatial_batchnorm_backward(dq, bn_cache)
    dx, dw, db = conv_backward_fast(dp, conv_cache)
    N = dgamma.shape[0]
    dbn = np.zeros((2, N))
    dbn[0, :] = dgamma
    dbn[1, :] = dbeta
    return dx, dw, db, dbn


def conv_bn_relu_pool_fwd(x, w, b, bn, conv_param, bn_params, pool_param):
    p, conv_cache = conv_forward_fast(x, w, b, conv_param)
    q, bn_cache = spatial_batchnorm_forward(p, bn[0, :], bn[1, :], bn_params)
    r, relu_cache = relu_forward(q)
    out, pool_cache = max_pool_forward_fast(r, pool_param)
    cache = (conv_cache, bn_cache, relu_cache, pool_cache)
    return out, cache

def conv_bn_relu_pool_bkwd(dout, cache):
    conv_cache, bn_cache, relu_cache, pool_cache = cache
    dr = max_pool_backward_fast(dout, pool_cache)
    dq = relu_backward(dr, relu_cache)
    dp, dgamma, dbeta = spatial_batchnorm_backward(dq, bn_cache)
    dx, dw, db = conv_backward_fast(dp, conv_cache)
    N = dgamma.shape[0]
    dbn = np.zeros((2, N))
    dbn[0, :] = dgamma
    dbn[1, :] = dbeta
    return dx, dw, db, dbn