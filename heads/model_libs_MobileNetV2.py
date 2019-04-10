import os
import sys

import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2

def check_if_exist(path):
    return os.path.exists(path)

def make_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)

def UnpackVariable(var, num):
  assert len > 0
  if type(var) is list and len(var) == num:
    return var
  else:
    ret = []
    if type(var) is list:
      assert len(var) == 1
      for i in xrange(0, num):
        ret.append(var[0])
    else:
      for i in xrange(0, num):
        ret.append(var)
    return ret

def ConvBNLayer2(net, from_layer, out_layer, use_bn, use_relu, num_output,
    kernel_size, pad, stride, dilation=1, use_scale=True, lr_mult=1,
    conv_prefix='', conv_postfix='', bn_prefix='', bn_postfix='_bn',
    scale_prefix='', scale_postfix='_scale', bias_prefix='', bias_postfix='_bias',
    **bn_params):
  if use_bn:
    # parameters for convolution layer with batchnorm.
    kwargs = {
        'param': [dict(lr_mult=lr_mult, decay_mult=1)],
        'weight_filler': dict(type='gaussian', std=0.01),
        'bias_term': False,
        }
    # parameters for scale bias layer after batchnorm.
    if use_scale:
      sb_kwargs = {
          'bias_term': True,
          }
    else:
      bias_kwargs = {
          'param': [dict(lr_mult=lr_mult, decay_mult=0)],
          'filler': dict(type='constant', value=0.0),
          }
  else:
    kwargs = {
        'param': [
            dict(lr_mult=lr_mult, decay_mult=1),
            dict(lr_mult=2 * lr_mult, decay_mult=0)],
        'weight_filler': dict(type='xavier'),
        'bias_filler': dict(type='constant', value=0)
        }

  conv_name = '{}{}{}'.format(conv_prefix, out_layer, conv_postfix)
  [kernel_h, kernel_w] = UnpackVariable(kernel_size, 2)
  [pad_h, pad_w] = UnpackVariable(pad, 2)
  [stride_h, stride_w] = UnpackVariable(stride, 2)
  if kernel_h == kernel_w:
    net[conv_name] = L.Convolution(net[from_layer], num_output=num_output,
        kernel_size=kernel_h, pad=pad_h, stride=stride_h, **kwargs)
  else:
    net[conv_name] = L.Convolution(net[from_layer], num_output=num_output,
        kernel_h=kernel_h, kernel_w=kernel_w, pad_h=pad_h, pad_w=pad_w,
        stride_h=stride_h, stride_w=stride_w, **kwargs)
  if dilation > 1:
    net.update(conv_name, {'dilation': dilation})
  if use_bn:
    bn_name = '{}{}{}'.format(bn_prefix, out_layer, bn_postfix)
    net[bn_name] = L.BatchNorm(net[conv_name], in_place=True)
    if use_scale:
      sb_name = '{}{}{}'.format(scale_prefix, out_layer, scale_postfix)
      net[sb_name] = L.Scale(net[bn_name], in_place=True, **sb_kwargs)
    else:
      bias_name = '{}{}{}'.format(bias_prefix, out_layer, bias_postfix)
      net[bias_name] = L.Bias(net[bn_name], in_place=True, **bias_kwargs)
  if use_relu:
    relu_name = '{}_relu'.format(conv_name) 
    net[relu_name] = L.ReLU(net[conv_name], in_place=True)

def ConvBNLayer(net, from_layer, out_layer, use_bn, use_relu, num_output,
    kernel_size, pad, stride, dilation=1, use_scale=True, lr_mult=1,
    conv_prefix='', conv_postfix='', bn_prefix='', bn_postfix='_bn',
    scale_prefix='', scale_postfix='_scale', bias_prefix='', bias_postfix='_bias',
    **bn_params):
  if use_bn:
    # parameters for convolution layer with batchnorm.
    kwargs = {
        'param': [dict(lr_mult=lr_mult, decay_mult=1)],
        'weight_filler': dict(type='gaussian', std=0.01),
        'bias_term': False,
        }
    # parameters for scale bias layer after batchnorm.
    if use_scale:
      sb_kwargs = {
          'bias_term': True,
          }
    else:
      bias_kwargs = {
          'param': [dict(lr_mult=lr_mult, decay_mult=0)],
          'filler': dict(type='constant', value=0.0),
          }
  else:
    kwargs = {
        'param': [
            dict(lr_mult=lr_mult, decay_mult=1),
            dict(lr_mult=2 * lr_mult, decay_mult=0)],
        'weight_filler': dict(type='xavier'),
        'bias_filler': dict(type='constant', value=0)
        }

  #conv_name = '{}{}{}'.format(conv_prefix, out_layer, conv_postfix)
  conv_name = '{}{}'.format(conv_prefix, out_layer)
  [kernel_h, kernel_w] = UnpackVariable(kernel_size, 2)
  [pad_h, pad_w] = UnpackVariable(pad, 2)
  [stride_h, stride_w] = UnpackVariable(stride, 2)
  if kernel_h == kernel_w:
    net[conv_name] = L.Convolution(net[from_layer], num_output=num_output,
        kernel_size=kernel_h, pad=pad_h, stride=stride_h, **kwargs)
  else:
    net[conv_name] = L.Convolution(net[from_layer], num_output=num_output,
        kernel_h=kernel_h, kernel_w=kernel_w, pad_h=pad_h, pad_w=pad_w,
        stride_h=stride_h, stride_w=stride_w, **kwargs)
  if dilation > 1:
    net.update(conv_name, {'dilation': dilation})
  if use_bn:
    bn_name = '{}{}{}'.format(bn_prefix, out_layer, bn_postfix)
    net[bn_name] = L.BatchNorm(net[conv_name], in_place=True)
    if use_scale:
      sb_name = '{}{}{}'.format(scale_prefix, out_layer, scale_postfix)
      net[sb_name] = L.Scale(net[bn_name], in_place=True, **sb_kwargs)
    else:
      bias_name = '{}{}{}'.format(bias_prefix, out_layer, bias_postfix)
      net[bias_name] = L.Bias(net[bn_name], in_place=True, **bias_kwargs)
  if use_relu:
    #relu_name = '{}_relu'.format(conv_name)
    relu_name = 'relu{}'.format(conv_postfix)
    net[relu_name] = L.ReLU(net[conv_name], in_place=True)

def DWConvBNLayer(net, from_layer, out_layer, use_bn, use_relu, num_output, group,
    kernel_size, pad, stride, dilation=1, use_scale=True, lr_mult=1,
    conv_prefix='', conv_postfix='', bn_prefix='', bn_postfix='_bn',
    scale_prefix='', scale_postfix='_scale', bias_prefix='', bias_postfix='_bias',
    **bn_params):
  if use_bn:
    # parameters for convolution layer with batchnorm.
    kwargs = {
        'param': [dict(lr_mult=lr_mult, decay_mult=1)],
        'weight_filler': dict(type='gaussian', std=0.01),
        'bias_term': False,
        }
    # parameters for scale bias layer after batchnorm.
    if use_scale:
      sb_kwargs = {
          'bias_term': True,
          }
    else:
      bias_kwargs = {
          'param': [dict(lr_mult=lr_mult, decay_mult=0)],
          'filler': dict(type='constant', value=0.0),
          }
  else:
    kwargs = {
        'param': [
            dict(lr_mult=lr_mult, decay_mult=1),
            dict(lr_mult=2 * lr_mult, decay_mult=0)],
        'weight_filler': dict(type='xavier'),
        'bias_filler': dict(type='constant', value=0)
        }

  conv_name = '{}{}'.format(conv_prefix, out_layer)
  [kernel_h, kernel_w] = UnpackVariable(kernel_size, 2)
  [pad_h, pad_w] = UnpackVariable(pad, 2)
  [stride_h, stride_w] = UnpackVariable(stride, 2)
  if kernel_h == kernel_w:
    #net[conv_name] = L.ConvolutionDepthwise(net[from_layer], num_output=num_output, group=group,
    #    kernel_size=kernel_h, pad=pad_h, stride=stride_h, **kwargs)
    net[conv_name] = L.ConvolutionDepthwise(net[from_layer],
        convolution_param = dict(num_output=num_output, group=group, pad=pad_h, kernel_size=kernel_h, stride=stride_h, bias_term=False, 
        weight_filler = dict(type='gaussian', std=0.01)), param=[dict(lr_mult=lr_mult, decay_mult=1)])
  else:
    #net[conv_name] = L.ConvolutionDepthwise(net[from_layer], num_output=num_output, group=group,
    #    kernel_h=kernel_h, kernel_w=kernel_w, pad_h=pad_h, pad_w=pad_w,
    #    stride_h=stride_h, stride_w=stride_w, **kwargs)
    net[conv_name] = L.ConvolutionDepthwise(net[from_layer],
        convolution_param = dict(num_output=num_output, group=group, kernel_h=kernel_h, 
        kernel_w=kernel_w, pad_h=pad_h, pad_w=pad_w, stride_h=stride_h, stride_w=stride_w, bias_term=False, 
        weight_filler = dict(type='gaussian', std=0.01)), param=[dict(lr_mult=lr_mult, decay_mult=1)])
  if dilation > 1:
    net.update(conv_name, {'dilation': dilation})
  if use_bn:
    bn_name = '{}{}{}'.format(bn_prefix, out_layer, bn_postfix)
    net[bn_name] = L.BatchNorm(net[conv_name], in_place=True)
    if use_scale:
      sb_name = '{}{}{}'.format(scale_prefix, out_layer, scale_postfix)
      net[sb_name] = L.Scale(net[bn_name], in_place=True, **sb_kwargs)
    else:
      bias_name = '{}{}{}'.format(bias_prefix, out_layer, bias_postfix)
      net[bias_name] = L.Bias(net[bn_name], in_place=True, **bias_kwargs)
  if use_relu:
    #relu_name = '{}_relu'.format(conv_name)
    relu_name = 'relu{}'.format(conv_postfix)
    net[relu_name] = L.ReLU(net[conv_name], in_place=True)

def DeconvBNLayer(net, from_layer, out_layer, use_bn, use_relu, num_output,
    kernel_size, pad, stride, use_scale=True, lr_mult=1, deconv_prefix='', deconv_postfix='',
    bn_prefix='', bn_postfix='_bn', scale_prefix='', scale_postfix='_scale', bias_prefix='',
    bias_postfix='_bias', **bn_params):
  if use_bn:
    kwargs = {
        'param': [dict(lr_mult=lr_mult, decay_mult=1)],
        'weight_filler': dict(type='gaussian', std=0.01),
        'bias_term': False,
        }
    # parameters for scale bias layer after batchnorm.
    if use_scale:
      sb_kwargs = {
          'bias_term': True,
          }
    else:
      bias_kwargs = {
          'param': [dict(lr_mult=lr_mult, decay_mult=0)],
          'filler': dict(type='constant', value=0.0),
          }
  else:
    kwargs = {
        'param': [
            dict(lr_mult=lr_mult, decay_mult=1),
            dict(lr_mult=2 * lr_mult, decay_mult=0)],
        'weight_filler': dict(type='xavier'),
        'bias_filler': dict(type='constant', value=0)
        }
  deconv_name = '{}{}{}'.format(deconv_prefix, out_layer, deconv_postfix)
  net[deconv_name] = L.Deconvolution(net[from_layer], num_output=num_output,
      kernel_size=kernel_size, pad=pad, stride=stride, **kwargs)

  if use_bn:
      bn_name = '{}{}{}'.format(bn_prefix, out_layer, bn_postfix)
      net[bn_name] = L.BatchNorm(net[deconv_name], in_place=True)
      if use_scale:
          sb_name = '{}{}{}'.format(scale_prefix, out_layer, scale_postfix)
          net[sb_name] = L.Scale(net[bn_name], in_place=True, **sb_kwargs)
      else:
          bias_name = '{}{}{}'.format(bias_prefix, out_layer, bias_postfix)
          net[bias_name] = L.Bias(net[bn_name], in_place=True, **bias_kwargs)

  if use_relu:
      relu_name = '{}_relu'.format(deconv_name)
      net[relu_name] = L.ReLU(net[deconv_name], in_place=True)

def EltwiseLayer(net, from_layer, out_layer):
  elt_name = out_layer
  net[elt_name] = L.Eltwise(net[from_layer[0]], net[from_layer[1]])


def ResBody(net, from_layer, block_name, out2a, out2b, out2c, stride, use_branch1, dilation=1, **bn_param):
  # ResBody(net, 'pool1', '2a', 64, 64, 256, 1, True)

  conv_prefix = 'res{}_'.format(block_name)
  conv_postfix = ''
  bn_prefix = 'bn{}_'.format(block_name)
  bn_postfix = ''
  scale_prefix = 'scale{}_'.format(block_name)
  scale_postfix = ''
  use_scale = True

  if use_branch1:
    branch_name = 'branch1'
    ConvBNLayer(net, from_layer, branch_name, use_bn=True, use_relu=False,
        num_output=out2c, kernel_size=1, pad=0, stride=stride, use_scale=use_scale,
        conv_prefix=conv_prefix, conv_postfix=conv_postfix,
        bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)
    branch1 = '{}{}'.format(conv_prefix, branch_name)
  else:
    branch1 = from_layer

  branch_name = 'branch2a'
  ConvBNLayer(net, from_layer, branch_name, use_bn=True, use_relu=True,
      num_output=out2a, kernel_size=1, pad=0, stride=stride, use_scale=use_scale,
      conv_prefix=conv_prefix, conv_postfix=conv_postfix,
      bn_prefix=bn_prefix, bn_postfix=bn_postfix,
      scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)
  out_name = '{}{}'.format(conv_prefix, branch_name)

  branch_name = 'branch2b'
  if dilation == 1:
    ConvBNLayer(net, out_name, branch_name, use_bn=True, use_relu=True,
        num_output=out2b, kernel_size=3, pad=1, stride=1, use_scale=use_scale,
        conv_prefix=conv_prefix, conv_postfix=conv_postfix,
        bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)
  else:
    pad = int((3 + (dilation - 1) * 2) - 1) / 2
    ConvBNLayer(net, out_name, branch_name, use_bn=True, use_relu=True,
        num_output=out2b, kernel_size=3, pad=pad, stride=1, use_scale=use_scale,
        dilation=dilation, conv_prefix=conv_prefix, conv_postfix=conv_postfix,
        bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)
  out_name = '{}{}'.format(conv_prefix, branch_name)

  branch_name = 'branch2c'
  ConvBNLayer(net, out_name, branch_name, use_bn=True, use_relu=False,
      num_output=out2c, kernel_size=1, pad=0, stride=1, use_scale=use_scale,
      conv_prefix=conv_prefix, conv_postfix=conv_postfix,
      bn_prefix=bn_prefix, bn_postfix=bn_postfix,
      scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)
  branch2 = '{}{}'.format(conv_prefix, branch_name)

  res_name = 'res{}'.format(block_name)
  net[res_name] = L.Eltwise(net[branch1], net[branch2])
  relu_name = '{}_relu'.format(res_name)
  net[relu_name] = L.ReLU(net[res_name], in_place=True)


def InceptionTower(net, from_layer, tower_name, layer_params, **bn_param):
  use_scale = False
  for param in layer_params:
    tower_layer = '{}/{}'.format(tower_name, param['name'])
    del param['name']
    if 'pool' in tower_layer:
      net[tower_layer] = L.Pooling(net[from_layer], **param)
    else:
      param.update(bn_param)
      ConvBNLayer(net, from_layer, tower_layer, use_bn=True, use_relu=True,
          use_scale=use_scale, **param)
    from_layer = tower_layer
  return net[from_layer]

def CreateAnnotatedDataLayer(source, batch_size=32, backend=P.Data.LMDB,
        output_label=True, train=True, label_map_file='', anno_type=None,
        transform_param={}, batch_sampler=[{}]):
    if train:
        kwargs = {
                'include': dict(phase=caffe_pb2.Phase.Value('TRAIN')),
                'transform_param': transform_param,
                }
    else:
        kwargs = {
                'include': dict(phase=caffe_pb2.Phase.Value('TEST')),
                'transform_param': transform_param,
                }
    ntop = 1
    if output_label:
        ntop = 2
    annotated_data_param = {
        'label_map_file': label_map_file,
        'batch_sampler': batch_sampler,
        }
    if anno_type is not None:
        annotated_data_param.update({'anno_type': anno_type})
    return L.AnnotatedData(name="data", annotated_data_param=annotated_data_param,
        data_param=dict(batch_size=batch_size, backend=backend, source=source),
        ntop=ntop, **kwargs)

def MobileNetV2Body(net, from_layer, use_final_pool=True, **bn_param):
    conv_prefix = ''
    conv_postfix = ''
    bn_prefix = ''    
    bn_postfix = '/bn'    
    scale_prefix = ''
    scale_postfix = '/scale'

    kwargs = {
        'param': [dict(lr_mult=1, decay_mult=1)],
        'weight_filler': dict(type='gaussian', std=0.01),
        'bias_term': False,
        }

    # Convoution 1
    ConvBNLayer(net, from_layer, 'conv1', use_bn=True, use_relu=True, num_output=32, kernel_size=3, pad=1, stride=2,
        conv_prefix=conv_prefix, conv_postfix='1', bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)
    
    # Depthwise convolution 2_1
    ConvBNLayer(net, 'conv1', 'conv2_1/expand', use_bn=True, use_relu=True, num_output=32, kernel_size=1, pad=0, stride=1,
        conv_prefix=conv_prefix, conv_postfix='2_1/expand', bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)
    DWConvBNLayer(net, 'conv2_1/expand', 'conv2_1/dwise', use_bn=True, use_relu=True, num_output=32, group=32, kernel_size=3, pad=1, stride=1,
        conv_prefix=conv_prefix, conv_postfix='2_1/dwise', bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)    
    ConvBNLayer(net, 'conv2_1/dwise', 'conv2_1/linear', use_bn=True, use_relu=False, num_output=16, kernel_size=1, pad=0, stride=1,
        conv_prefix=conv_prefix, conv_postfix='2_1/linear', bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)

    # Dpethwise convolution 2_2
    ConvBNLayer(net, 'conv2_1/linear', 'conv2_2/expand', use_bn=True, use_relu=True, num_output=96, kernel_size=1, pad=0, stride=1,
        conv_prefix=conv_prefix, conv_postfix='2_2/expand', bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)
    DWConvBNLayer(net, 'conv2_2/expand', 'conv2_2/dwise', use_bn=True, use_relu=True, num_output=96, group=96, kernel_size=3, pad=1, stride=2,
        conv_prefix=conv_prefix, conv_postfix='2_2/dwise', bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)    
    ConvBNLayer(net, 'conv2_2/dwise', 'conv2_2/linear', use_bn=True, use_relu=False, num_output=24, kernel_size=1, pad=0, stride=1,
        conv_prefix=conv_prefix, conv_postfix='2_2/linear', bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)

    # Right branch of depthwise convolution 3_1
    ConvBNLayer(net, 'conv2_2/linear', 'conv3_1/expand', use_bn=True, use_relu=True, num_output=144, kernel_size=1, pad=0, stride=1,
        conv_prefix=conv_prefix, conv_postfix='3_1/expand', bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)
    DWConvBNLayer(net, 'conv3_1/expand', 'conv3_1/dwise', use_bn=True, use_relu=True, num_output=144, group=144, kernel_size=3, pad=1, stride=1,
        conv_prefix=conv_prefix, conv_postfix='3_1/dwise', bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)    
    ConvBNLayer(net, 'conv3_1/dwise', 'conv3_1/linear', use_bn=True, use_relu=False, num_output=24, kernel_size=1, pad=0, stride=1,
        conv_prefix=conv_prefix, conv_postfix='3_1/linear', bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)
    net.block_3_1 = L.Eltwise(net['conv2_2/linear'], net['conv3_1/linear'])

    # Depthwise convolution 3_2
    ConvBNLayer(net, 'block_3_1', 'conv3_2/expand', use_bn=True, use_relu=True, num_output=144, kernel_size=1, pad=0, stride=1,
        conv_prefix=conv_prefix, conv_postfix='3_2/expand', bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)
    DWConvBNLayer(net, 'conv3_2/expand', 'conv3_2/dwise', use_bn=True, use_relu=True, num_output=144, group=144, kernel_size=3, pad=1, stride=2,
        conv_prefix=conv_prefix, conv_postfix='3_2/dwise', bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)    
    ConvBNLayer(net, 'conv3_2/dwise', 'conv3_2/linear', use_bn=True, use_relu=False, num_output=32, kernel_size=1, pad=0, stride=1,
        conv_prefix=conv_prefix, conv_postfix='3_2/linear', bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)

    # Right branch of depthwise convolution 4_1
    ConvBNLayer(net, 'conv3_2/linear', 'conv4_1/expand', use_bn=True, use_relu=True, num_output=192, kernel_size=1, pad=0, stride=1,
        conv_prefix=conv_prefix, conv_postfix='4_1/expand', bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)
    DWConvBNLayer(net, 'conv4_1/expand', 'conv4_1/dwise', use_bn=True, use_relu=True, num_output=192, group=192, kernel_size=3, pad=1, stride=1,
        conv_prefix=conv_prefix, conv_postfix='4_1/dwise', bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)    
    ConvBNLayer(net, 'conv4_1/dwise', 'conv4_1/linear', use_bn=True, use_relu=False, num_output=32, kernel_size=1, pad=0, stride=1,
        conv_prefix=conv_prefix, conv_postfix='4_1/linear', bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)
    net.block_4_1 = L.Eltwise(net['conv3_2/linear'], net['conv4_1/linear'])

    # Right branch of depthwise convolution 4_2
    ConvBNLayer(net, 'block_4_1', 'conv4_2/expand', use_bn=True, use_relu=True, num_output=192, kernel_size=1, pad=0, stride=1,
        conv_prefix=conv_prefix, conv_postfix='4_2/expand', bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)
    DWConvBNLayer(net, 'conv4_2/expand', 'conv4_2/dwise', use_bn=True, use_relu=True, num_output=192, group=192, kernel_size=3, pad=1, stride=1,
        conv_prefix=conv_prefix, conv_postfix='4_2/dwise', bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)    
    ConvBNLayer(net, 'conv4_2/dwise', 'conv4_2/linear', use_bn=True, use_relu=False, num_output=32, kernel_size=1, pad=0, stride=1,
        conv_prefix=conv_prefix, conv_postfix='4_2/linear', bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)
    net.block_4_2 = L.Eltwise(net['block_4_1'], net['conv4_2/linear'])

    # Right branch of depthwise convolution 4_3
    ConvBNLayer(net, 'block_4_2', 'conv4_3/expand', use_bn=True, use_relu=True, num_output=192, kernel_size=1, pad=0, stride=1,
        conv_prefix=conv_prefix, conv_postfix='4_3/expand', bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)
    DWConvBNLayer(net, 'conv4_3/expand', 'conv4_3/dwise', use_bn=True, use_relu=True, num_output=192, group=192, kernel_size=3, pad=1, stride=1,
        conv_prefix=conv_prefix, conv_postfix='4_3/dwise', bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)    
    ConvBNLayer(net, 'conv4_3/dwise', 'conv4_3/linear', use_bn=True, use_relu=False, num_output=64, kernel_size=1, pad=0, stride=1,
        conv_prefix=conv_prefix, conv_postfix='4_3/linear', bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)

    # Right branch of depthwise convolution 4_4
    ConvBNLayer(net, 'conv4_3/linear', 'conv4_4/expand', use_bn=True, use_relu=True, num_output=384, kernel_size=1, pad=0, stride=1,
        conv_prefix=conv_prefix, conv_postfix='4_4/expand', bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)
    DWConvBNLayer(net, 'conv4_4/expand', 'conv4_4/dwise', use_bn=True, use_relu=True, num_output=384, group=384, kernel_size=3, pad=1, stride=1,
        conv_prefix=conv_prefix, conv_postfix='4_4/dwise', bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)    
    ConvBNLayer(net, 'conv4_4/dwise', 'conv4_4/linear', use_bn=True, use_relu=False, num_output=64, kernel_size=1, pad=0, stride=1,
        conv_prefix=conv_prefix, conv_postfix='4_4/linear', bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)
    net.block_4_4 = L.Eltwise(net['conv4_3/linear'], net['conv4_4/linear'])

    # Right branch of depthwise convolution 4_5
    ConvBNLayer(net, 'block_4_4', 'conv4_5/expand', use_bn=True, use_relu=True, num_output=384, kernel_size=1, pad=0, stride=1,
        conv_prefix=conv_prefix, conv_postfix='4_5/expand', bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)
    DWConvBNLayer(net, 'conv4_5/expand', 'conv4_5/dwise', use_bn=True, use_relu=True, num_output=384, group=384, kernel_size=3, pad=1, stride=1,
        conv_prefix=conv_prefix, conv_postfix='4_5/dwise', bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)    
    ConvBNLayer(net, 'conv4_5/dwise', 'conv4_5/linear', use_bn=True, use_relu=False, num_output=64, kernel_size=1, pad=0, stride=1,
        conv_prefix=conv_prefix, conv_postfix='4_5/linear', bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)
    net.block_4_5 = L.Eltwise(net['block_4_4'], net['conv4_5/linear'])

    # Right branch of depthwise convolution 4_6
    ConvBNLayer(net, 'block_4_5', 'conv4_6/expand', use_bn=True, use_relu=True, num_output=384, kernel_size=1, pad=0, stride=1,
        conv_prefix=conv_prefix, conv_postfix='4_6/expand', bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)
    DWConvBNLayer(net, 'conv4_6/expand', 'conv4_6/dwise', use_bn=True, use_relu=True, num_output=384, group=384, kernel_size=3, pad=1, stride=1,
        conv_prefix=conv_prefix, conv_postfix='4_6/dwise', bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)    
    ConvBNLayer(net, 'conv4_6/dwise', 'conv4_6/linear', use_bn=True, use_relu=False, num_output=64, kernel_size=1, pad=0, stride=1,
        conv_prefix=conv_prefix, conv_postfix='4_6/linear', bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)
    net.block_4_6 = L.Eltwise(net['block_4_5'], net['conv4_6/linear'])

    # Depthwise convolution 4_7
    ConvBNLayer(net, 'block_4_6', 'conv4_7/expand', use_bn=True, use_relu=True, num_output=384, kernel_size=1, pad=0, stride=1,
        conv_prefix=conv_prefix, conv_postfix='4_7/expand', bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)
    DWConvBNLayer(net, 'conv4_7/expand', 'conv4_7/dwise', use_bn=True, use_relu=True, num_output=384, group=384, kernel_size=3, pad=1, stride=2,
        conv_prefix=conv_prefix, conv_postfix='4_7/dwise', bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)    
    ConvBNLayer(net, 'conv4_7/dwise', 'conv4_7/linear', use_bn=True, use_relu=False, num_output=96, kernel_size=1, pad=0, stride=1,
        conv_prefix=conv_prefix, conv_postfix='4_7/linear', bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)

    # Right branch of depthwise convolution 5_1
    ConvBNLayer(net, 'conv4_7/linear', 'conv5_1/expand', use_bn=True, use_relu=True, num_output=576, kernel_size=1, pad=0, stride=1,
        conv_prefix=conv_prefix, conv_postfix='5_1/expand', bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)
    DWConvBNLayer(net, 'conv5_1/expand', 'conv5_1/dwise', use_bn=True, use_relu=True, num_output=576, group=576, kernel_size=3, pad=1, stride=1,
        conv_prefix=conv_prefix, conv_postfix='5_1/dwise', bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)    
    ConvBNLayer(net, 'conv5_1/dwise', 'conv5_1/linear', use_bn=True, use_relu=False, num_output=96, kernel_size=1, pad=0, stride=1,
        conv_prefix=conv_prefix, conv_postfix='5_1/linear', bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)
    net.block_5_1 = L.Eltwise(net['conv4_7/linear'], net['conv5_1/linear'])

    # Right branch of depthwise convolution 5_2
    ConvBNLayer(net, 'block_5_1', 'conv5_2/expand', use_bn=True, use_relu=True, num_output=576, kernel_size=1, pad=0, stride=1,
        conv_prefix=conv_prefix, conv_postfix='5_2/expand', bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)
    DWConvBNLayer(net, 'conv5_2/expand', 'conv5_2/dwise', use_bn=True, use_relu=True, num_output=576, group=576, kernel_size=3, pad=1, stride=1,
        conv_prefix=conv_prefix, conv_postfix='5_2/dwise', bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)    
    ConvBNLayer(net, 'conv5_2/dwise', 'conv5_2/linear', use_bn=True, use_relu=False, num_output=96, kernel_size=1, pad=0, stride=1,
        conv_prefix=conv_prefix, conv_postfix='5_2/linear', bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)
    net.block_5_2 = L.Eltwise(net['block_5_1'], net['conv5_2/linear'])

    # Depthwise convolution 5_3
    ConvBNLayer(net, 'block_5_2', 'conv5_3/expand', use_bn=True, use_relu=True, num_output=576, kernel_size=1, pad=0, stride=1,
        conv_prefix=conv_prefix, conv_postfix='5_3/expand', bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)
    DWConvBNLayer(net, 'conv5_3/expand', 'conv5_3/dwise', use_bn=True, use_relu=True, num_output=576, group=576, kernel_size=3, pad=1, stride=2,
        conv_prefix=conv_prefix, conv_postfix='5_3/dwise', bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)    
    ConvBNLayer(net, 'conv5_3/dwise', 'conv5_3/linear', use_bn=True, use_relu=False, num_output=160, kernel_size=1, pad=0, stride=1,
        conv_prefix=conv_prefix, conv_postfix='5_3/linear', bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)

    # Right branch of depthwise convolution 6_1
    ConvBNLayer(net, 'conv5_3/linear', 'conv6_1/expand', use_bn=True, use_relu=True, num_output=960, kernel_size=1, pad=0, stride=1,
        conv_prefix=conv_prefix, conv_postfix='6_1/expand', bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)
    DWConvBNLayer(net, 'conv6_1/expand', 'conv6_1/dwise', use_bn=True, use_relu=True, num_output=960, group=960, kernel_size=3, pad=1, stride=1,
        conv_prefix=conv_prefix, conv_postfix='6_1/dwise', bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)    
    ConvBNLayer(net, 'conv6_1/dwise', 'conv6_1/linear', use_bn=True, use_relu=False, num_output=160, kernel_size=1, pad=0, stride=1,
        conv_prefix=conv_prefix, conv_postfix='6_1/linear', bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)
    net.block_6_1 = L.Eltwise(net['conv5_3/linear'], net['conv6_1/linear'])

    # Right branch of depthwise convolution 6_2
    ConvBNLayer(net, 'block_6_1', 'conv6_2/expand', use_bn=True, use_relu=True, num_output=960, kernel_size=1, pad=0, stride=1,
        conv_prefix=conv_prefix, conv_postfix='6_2/expand', bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)
    DWConvBNLayer(net, 'conv6_2/expand', 'conv6_2/dwise', use_bn=True, use_relu=True, num_output=960, group=960, kernel_size=3, pad=1, stride=1,
        conv_prefix=conv_prefix, conv_postfix='6_2/dwise', bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)    
    ConvBNLayer(net, 'conv6_2/dwise', 'conv6_2/linear', use_bn=True, use_relu=False, num_output=160, kernel_size=1, pad=0, stride=1,
        conv_prefix=conv_prefix, conv_postfix='6_2/linear', bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)
    net.block_6_2 = L.Eltwise(net['block_6_1'], net['conv6_2/linear'])

    # Depthwise convolution 6_3
    ConvBNLayer(net, 'block_6_2', 'conv6_3/expand', use_bn=True, use_relu=True, num_output=960, kernel_size=1, pad=0, stride=1,
        conv_prefix=conv_prefix, conv_postfix='6_3/expand', bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)
    DWConvBNLayer(net, 'conv6_3/expand', 'conv6_3/dwise', use_bn=True, use_relu=True, num_output=960, group=960, kernel_size=3, pad=1, stride=1,
        conv_prefix=conv_prefix, conv_postfix='6_3/dwise', bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)    
    ConvBNLayer(net, 'conv6_3/dwise', 'conv6_3/linear', use_bn=True, use_relu=False, num_output=320, kernel_size=1, pad=0, stride=1,
        conv_prefix=conv_prefix, conv_postfix='6_3/linear', bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)

    # Convolution 6_4
    ConvBNLayer(net, 'conv6_3/linear', 'conv6_4', use_bn=True, use_relu=True, num_output=1280, kernel_size=1, pad=0, stride=1,
        conv_prefix=conv_prefix, conv_postfix='6_4', bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)

    if use_final_pool:
      net.final_pool = L.Pooling(net['conv6_4'], pool=P.Pooling.AVE, global_pooling=True)

    return net

def CreateMultiBoxHead(net, data_layer="data", num_classes=[], from_layers=[],
        use_objectness=False, normalizations=[], use_batchnorm=True, lr_mult=1,
        use_scale=True, min_sizes=[], max_sizes=[], prior_variance = [0.1],
        aspect_ratios=[], steps=[], img_height=0, img_width=0, share_location=True,
        flip=True, clip=True, offset=0.5, inter_layer_depth=[], kernel_size=1, pad=0, prefix='',
        conf_postfix='', loc_postfix='', **bn_param):
    assert num_classes, "must provide num_classes"
    assert num_classes > 0, "num_classes must be positive number"
    if normalizations:
        assert len(from_layers) == len(normalizations), "from_layers and normalizations should have same length"
    assert len(from_layers) == len(min_sizes), "from_layers and min_sizes should have same length"
    if max_sizes:
        assert len(from_layers) == len(max_sizes), "from_layers and max_sizes should have same length"
    if aspect_ratios:
        assert len(from_layers) == len(aspect_ratios), "from_layers and aspect_ratios should have same length"
    if steps:
        assert len(from_layers) == len(steps), "from_layers and steps should have same length"
    net_layers = net.keys()
    assert data_layer in net_layers, "data_layer is not in net's layers"
    if inter_layer_depth:
        assert len(from_layers) == len(inter_layer_depth), "from_layers and inter_layer_depth should have same length"

    num = len(from_layers)
    priorbox_layers = []
    loc_layers = []
    conf_layers = []
    objectness_layers = []
    for i in range(0, num):
        from_layer = from_layers[i]

        # Get the normalize value.
        if normalizations:
            if normalizations[i] != -1:
                norm_name = "{}_norm".format(from_layer)
                net[norm_name] = L.Normalize(net[from_layer], scale_filler=dict(type="constant", value=normalizations[i]),
                    across_spatial=False, channel_shared=False)
                from_layer = norm_name

        # Add intermediate layers.
        if inter_layer_depth:
            if inter_layer_depth[i] > 0:
                inter_name = "{}_inter".format(from_layer)
                ConvBNLayer(net, from_layer, inter_name, use_bn=use_batchnorm, use_relu=True, lr_mult=lr_mult,
                      num_output=inter_layer_depth[i], kernel_size=3, pad=1, stride=1, **bn_param)
                from_layer = inter_name

        # Estimate number of priors per location given provided parameters.
        min_size = min_sizes[i]
        if type(min_size) is not list:
            min_size = [min_size]
        aspect_ratio = []
        if len(aspect_ratios) > i:
            aspect_ratio = aspect_ratios[i]
            if type(aspect_ratio) is not list:
                aspect_ratio = [aspect_ratio]
        max_size = []
        if len(max_sizes) > i:
            max_size = max_sizes[i]
            if type(max_size) is not list:
                max_size = [max_size]
            if max_size:
                assert len(max_size) == len(min_size), "max_size and min_size should have same length."
        if max_size:
            num_priors_per_location = (2 + len(aspect_ratio)) * len(min_size)
        else:
            num_priors_per_location = (1 + len(aspect_ratio)) * len(min_size)
        if flip:
            num_priors_per_location += len(aspect_ratio) * len(min_size)
        step = []
        if len(steps) > i:
            step = steps[i]

        # Create location prediction layer.
        name = "{}_mbox_loc{}".format(from_layer, loc_postfix)
        num_loc_output = num_priors_per_location * 4;
        if not share_location:
            num_loc_output *= num_classes
        ConvBNLayer(net, from_layer, name, use_bn=use_batchnorm, use_relu=False, lr_mult=lr_mult,
            num_output=num_loc_output, kernel_size=kernel_size, pad=pad, stride=1, **bn_param)
        permute_name = "{}_perm".format(name)
        net[permute_name] = L.Permute(net[name], order=[0, 2, 3, 1])
        flatten_name = "{}_flat".format(name)
        net[flatten_name] = L.Flatten(net[permute_name], axis=1)
        loc_layers.append(net[flatten_name])

        # Create confidence prediction layer.
        name = "{}_mbox_conf{}".format(from_layer, conf_postfix)
        num_conf_output = num_priors_per_location * num_classes;
        ConvBNLayer(net, from_layer, name, use_bn=use_batchnorm, use_relu=False, lr_mult=lr_mult,
            num_output=num_conf_output, kernel_size=kernel_size, pad=pad, stride=1, **bn_param)
        permute_name = "{}_perm".format(name)
        net[permute_name] = L.Permute(net[name], order=[0, 2, 3, 1])
        flatten_name = "{}_flat".format(name)
        net[flatten_name] = L.Flatten(net[permute_name], axis=1)
        conf_layers.append(net[flatten_name])

        # Create prior generation layer.
        name = "{}_mbox_priorbox".format(from_layer)        
        net[name] = L.PriorBox(net[from_layer], net[data_layer], min_size=min_size,          
                clip=clip, variance=prior_variance, offset=offset)
        
        if max_size:
            net.update(name, {'max_size': max_size})
        if aspect_ratio:
            net.update(name, {'aspect_ratio': aspect_ratio, 'flip': flip})
        if step:
            net.update(name, {'step': step})
        if img_height != 0 and img_width != 0:
            if img_height == img_width:
                net.update(name, {'img_size': img_height})
            else:
                net.update(name, {'img_h': img_height, 'img_w': img_width})
        priorbox_layers.append(net[name])

        # Create objectness prediction layer.
        if use_objectness:
            name = "{}_mbox_objectness".format(from_layer)
            num_obj_output = num_priors_per_location * 2;
            ConvBNLayer(net, from_layer, name, use_bn=use_batchnorm, use_relu=False, lr_mult=lr_mult,
                num_output=num_obj_output, kernel_size=kernel_size, pad=pad, stride=1, **bn_param)
            permute_name = "{}_perm".format(name)
            net[permute_name] = L.Permute(net[name], order=[0, 2, 3, 1])
            flatten_name = "{}_flat".format(name)
            net[flatten_name] = L.Flatten(net[permute_name], axis=1)
            objectness_layers.append(net[flatten_name])

    # Concatenate priorbox, loc, and conf layers.
    mbox_layers = []
    name = '{}{}'.format(prefix, "_loc")
    net[name] = L.Concat(*loc_layers, axis=1)
    mbox_layers.append(net[name])
    name = '{}{}'.format(prefix, "_conf")
    net[name] = L.Concat(*conf_layers, axis=1)
    mbox_layers.append(net[name])
    name = '{}{}'.format(prefix, "_priorbox")
    net[name] = L.Concat(*priorbox_layers, axis=2)
    mbox_layers.append(net[name])
    if use_objectness:
        name = '{}{}'.format(prefix, "_objectness")
        net[name] = L.Concat(*objectness_layers, axis=1)
        mbox_layers.append(net[name])

    return mbox_layers

def CreateRefineDetHead(net, data_layer="data", num_classes=[], from_layers=[], from_layers2=[],
        normalizations=[], use_batchnorm=True, lr_mult=1, min_sizes=[], max_sizes=[], prior_variance = [0.1],
        aspect_ratios=[], steps=[], img_height=0, img_width=0, share_location=True,
        flip=True, clip=True, offset=0.5, inter_layer_depth=[], kernel_size=1, pad=0,
        conf_postfix='', loc_postfix='', **bn_param):
    assert num_classes, "must provide num_classes"
    assert num_classes > 0, "num_classes must be positive number"
    if normalizations:
        assert len(from_layers) == len(normalizations), "from_layers and normalizations should have same length"
    assert len(from_layers) == len(min_sizes), "from_layers and min_sizes should have same length"
    if max_sizes:
        assert len(from_layers) == len(max_sizes), "from_layers and max_sizes should have same length"
    if aspect_ratios:
        assert len(from_layers) == len(aspect_ratios), "from_layers and aspect_ratios should have same length"
    if steps:
        assert len(from_layers) == len(steps), "from_layers and steps should have same length"
    net_layers = net.keys()
    assert data_layer in net_layers, "data_layer is not in net's layers"
    if inter_layer_depth:
        assert len(from_layers) == len(inter_layer_depth), "from_layers and inter_layer_depth should have same length"

    use_relu = True
    conv_prefix = ''
    conv_postfix = ''
    bn_prefix = ''
    bn_postfix = '/bn'
    scale_prefix = ''
    scale_postfix = '/scale'   

    kwargs = {
      'param': [dict(lr_mult=1, decay_mult=1)],
      'weight_filler': dict(type='gaussian', std=0.01),
      'bias_term': False,
      }
    kwargs2 = {
        'param': [dict(lr_mult=1, decay_mult=1)],
        'weight_filler': dict(type='gaussian', std=0.01),
      }
    kwargs_sb = {
        'axis': 0,
        'bias_term': False
      }

    prefix = 'arm'
    num_classes_rpn = 2
    num = len(from_layers)
    priorbox_layers = []
    loc_layers = []
    conf_layers = []
    for i in range(0, num):
        from_layer = from_layers[i]

        # Get the normalize value.
        if normalizations:
            if normalizations[i] != -1:
                norm_name = "{}_norm".format(from_layer)
                net[norm_name] = L.Normalize(net[from_layer], scale_filler=dict(type="constant", value=normalizations[i]),
                    across_spatial=False, channel_shared=False)
                from_layer = norm_name

        # Add intermediate layers.
        if inter_layer_depth:
            if inter_layer_depth[i] > 0:                
                
                # Inter layer from body to head
                inter_name = "{}_inter".format(from_layer)
                # Expand layer
                inter_expand = inter_name + '/expand'
                ConvBNLayer(net, from_layer, inter_expand, use_bn=True, use_relu=True, num_output=576, kernel_size=1, pad=0, stride=1,
                    conv_prefix=conv_prefix, conv_postfix=inter_expand, bn_prefix=bn_prefix, bn_postfix=bn_postfix,
                    scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)
                # Depthwise convolution layer
                inter_dwise = inter_name + '/dwise'
                DWConvBNLayer(net, inter_expand, inter_dwise, use_bn=True, use_relu=True, num_output=576, group=576, kernel_size=3, pad=1, stride=1,
                    conv_prefix=conv_prefix, conv_postfix=inter_dwise, bn_prefix=bn_prefix, bn_postfix=bn_postfix,
                    scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)
                # Linear layer
                inter_linear = inter_name + '/linear'
                ConvBNLayer(net, inter_dwise, inter_linear, use_bn=True, use_relu=False, num_output=96, kernel_size=1, pad=0, stride=1,
                    conv_prefix=conv_prefix, conv_postfix=inter_linear, bn_prefix=bn_prefix, bn_postfix=bn_postfix,
                    scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)
                # Bridge of rest of head
                from_layer = inter_linear 
                
        # Estimate number of priors per location given provided parameters.
        min_size = min_sizes[i]
        if type(min_size) is not list:
            min_size = [min_size]
        aspect_ratio = []
        if len(aspect_ratios) > i:
            aspect_ratio = aspect_ratios[i]
            if type(aspect_ratio) is not list:
                aspect_ratio = [aspect_ratio]
        max_size = []
        if len(max_sizes) > i:
            max_size = max_sizes[i]
            if type(max_size) is not list:
                max_size = [max_size]
            if max_size:
                assert len(max_size) == len(min_size), "max_size and min_size should have same length."
        if max_size:
            num_priors_per_location = (2 + len(aspect_ratio)) * len(min_size)
        else:
            num_priors_per_location = (1 + len(aspect_ratio)) * len(min_size)
        if flip:
            num_priors_per_location += len(aspect_ratio) * len(min_size)
        step = []
        if len(steps) > i:
            step = steps[i]
        # Create location prediction layer.
        name = "{}_mbox_loc{}".format(from_layer, loc_postfix)
        num_loc_output = num_priors_per_location * 4
        if not share_location:
            num_loc_output *= num_classes_rpn
        ConvBNLayer2(net, from_layer, name, use_bn=use_batchnorm, use_relu=False, lr_mult=lr_mult,
            num_output=num_loc_output, kernel_size=kernel_size, pad=pad, stride=1, **bn_param)
        permute_name = "{}_perm".format(name)
        net[permute_name] = L.Permute(net[name], order=[0, 2, 3, 1])
        flatten_name = "{}_flat".format(name)
        net[flatten_name] = L.Flatten(net[permute_name], axis=1)
        loc_layers.append(net[flatten_name])        

        # Create confidence prediction layer.
        name = "{}_mbox_conf{}".format(from_layer, conf_postfix)
        num_conf_output = num_priors_per_location * num_classes_rpn
        ConvBNLayer2(net, from_layer, name, use_bn=use_batchnorm, use_relu=False, lr_mult=lr_mult,
            num_output=num_conf_output, kernel_size=kernel_size, pad=pad, stride=1, **bn_param)
        permute_name = "{}_perm".format(name)
        net[permute_name] = L.Permute(net[name], order=[0, 2, 3, 1])
        flatten_name = "{}_flat".format(name)
        net[flatten_name] = L.Flatten(net[permute_name], axis=1)
        conf_layers.append(net[flatten_name])

        # Create prior generation layer.
        name = "{}_mbox_priorbox".format(from_layer)
        net[name] = L.PriorBox(net[from_layer], net[data_layer], min_size=min_size,
                clip=clip, variance=prior_variance, offset=offset)

        if max_size:
            net.update(name, {'max_size': max_size})
        if aspect_ratio:
            net.update(name, {'aspect_ratio': aspect_ratio, 'flip': flip})
        if step:
            net.update(name, {'step': step})
        if img_height != 0 and img_width != 0:
            if img_height == img_width:
                net.update(name, {'img_size': img_height})
            else:
                net.update(name, {'img_h': img_height, 'img_w': img_width})
        priorbox_layers.append(net[name])

    # Concatenate priorbox, loc, and conf layers.
    mbox_layers = []
    name = '{}{}'.format(prefix, "_loc")
    net[name] = L.Concat(*loc_layers, axis=1)
    mbox_layers.append(net[name])
    name = '{}{}'.format(prefix, "_conf")
    net[name] = L.Concat(*conf_layers, axis=1)
    mbox_layers.append(net[name])
    name = '{}{}'.format(prefix, "_priorbox")
    net[name] = L.Concat(*priorbox_layers, axis=2)
    mbox_layers.append(net[name])

    prefix = 'odm'
    num = len(from_layers2)
    loc_layers = []
    conf_layers = []
    for i in range(0, num):
        from_layer = from_layers2[i]

        # Get the normalize value.
        if normalizations:
            if normalizations[i] != -1:
                norm_name = "{}_norm".format(from_layer)
                net[norm_name] = L.Normalize(net[from_layer], scale_filler=dict(type="constant", value=normalizations[i]),
                    across_spatial=False, channel_shared=False)
                from_layer = norm_name

        # Add intermediate layers.
        if inter_layer_depth:
            if inter_layer_depth[i] > 0:
                
                # Inter layer from body to head
                inter_name = "{}_inter".format(from_layer)
                # Expand layer
                inter_expand = inter_name + '/expand'
                ConvBNLayer(net, from_layer, inter_expand, use_bn=True, use_relu=True, num_output=576, kernel_size=1, pad=0, stride=1,
                    conv_prefix=conv_prefix, conv_postfix=inter_expand, bn_prefix=bn_prefix, bn_postfix=bn_postfix,
                    scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)
                # Depthwise convolution layer
                inter_dwise = inter_name + '/dwise'
                DWConvBNLayer(net, inter_expand, inter_dwise, use_bn=True, use_relu=True, num_output=576, group=576, kernel_size=3, pad=1, stride=1,
                    conv_prefix=conv_prefix, conv_postfix=inter_dwise, bn_prefix=bn_prefix, bn_postfix=bn_postfix,
                    scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)
                # Linear layer
                inter_linear = inter_name + '/linear'
                ConvBNLayer(net, inter_dwise, inter_linear, use_bn=True, use_relu=False, num_output=96, kernel_size=1, pad=0, stride=1,
                    conv_prefix=conv_prefix, conv_postfix=inter_linear, bn_prefix=bn_prefix, bn_postfix=bn_postfix,
                    scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)
                # Bridge of rest of head
                from_layer = inter_linear

        # Estimate number of priors per location given provided parameters.
        min_size = min_sizes[i]
        if type(min_size) is not list:
            min_size = [min_size]
        aspect_ratio = []
        if len(aspect_ratios) > i:
            aspect_ratio = aspect_ratios[i]
            if type(aspect_ratio) is not list:
                aspect_ratio = [aspect_ratio]
        max_size = []
        if len(max_sizes) > i:
            max_size = max_sizes[i]
            if type(max_size) is not list:
                max_size = [max_size]
            if max_size:
                assert len(max_size) == len(min_size), "max_size and min_size should have same length."
        if max_size:
            num_priors_per_location = (2 + len(aspect_ratio)) * len(min_size)
        else:
            num_priors_per_location = (1 + len(aspect_ratio)) * len(min_size)
        if flip:
            num_priors_per_location += len(aspect_ratio) * len(min_size)

        # Create location prediction layer.
        name = "{}_mbox_loc{}".format(from_layer, loc_postfix)
        num_loc_output = num_priors_per_location * 4
        if not share_location:
            num_loc_output *= num_classes
        ConvBNLayer2(net, from_layer, name, use_bn=use_batchnorm, use_relu=False, lr_mult=lr_mult,
                    num_output=num_loc_output, kernel_size=kernel_size, pad=pad, stride=1, **bn_param)
        permute_name = "{}_perm".format(name)
        net[permute_name] = L.Permute(net[name], order=[0, 2, 3, 1])
        flatten_name = "{}_flat".format(name)
        net[flatten_name] = L.Flatten(net[permute_name], axis=1)
        loc_layers.append(net[flatten_name])

        # Create confidence prediction layer.
        name = "{}_mbox_conf{}".format(from_layer, conf_postfix)
        num_conf_output = num_priors_per_location * num_classes
        ConvBNLayer2(net, from_layer, name, use_bn=use_batchnorm, use_relu=False, lr_mult=lr_mult,
                    num_output=num_conf_output, kernel_size=kernel_size, pad=pad, stride=1, **bn_param)
        permute_name = "{}_perm".format(name)
        net[permute_name] = L.Permute(net[name], order=[0, 2, 3, 1])
        flatten_name = "{}_flat".format(name)
        net[flatten_name] = L.Flatten(net[permute_name], axis=1)
        conf_layers.append(net[flatten_name])


    # Concatenate priorbox, loc, and conf layers.
    name = '{}{}'.format(prefix, "_loc")
    net[name] = L.Concat(*loc_layers, axis=1)
    mbox_layers.append(net[name])
    name = '{}{}'.format(prefix, "_conf")
    net[name] = L.Concat(*conf_layers, axis=1)
    mbox_layers.append(net[name])    

    return mbox_layers
