import tensorflow as tf
import os, sys
from nnBase import nnBase
import json

default_dir = 'E:\\Python\\cloud\\network\\'

def create_json_template(filename):
    n_outs = [64, 64, None, 128, 128, None, 256, 256, 256, None, 512, 512, 512, None, 1024, 256]
    # Note that the name of 'layers' must contain '_',
    # and the letter after the last '_' must be integers,
    # indicating the unique index of the layer,
    # the integer will be used as keys to sort the layers.
    layers = [
        'conv_{}', 'conv_{}', 'pool_{}', 'conv_{}', 'conv_{}', 'pool_{}', 'conv_{}',
        'conv_{}', 'conv_{}', 'pool_{}', 'conv_{}', 'conv_{}', 'conv_{}', 'pool_{}',
        'dense_{}', 'dense_{}'
    ]
    layers = [item.format(it) for item, it in zip(layers, range(len(layers)))]
    # activation can be either 'relu' or 'lrelu'
    # loss can be either 'cross_entropy' or 'least_square'
    template = {
        'model_name': 'VGG',
        'layers': {ax: bx for ax, bx in zip(layers, n_outs)},
        'conv_kernel_size': 3,
        'conv_stride': 1,
        'pooling_type': 'max',
        'pooling_stride': 2,
        'use_batch_norm': True,
        'is_training': True,
        'activation': 'relu',
        'use_dropout': True,
        'keep_prob': 0.5,
        'num_classes': 10,
        'use_softmax': True,
        'loss': 'cross_entropy',
        'sparse_ys': True
    }
    with open(filename, 'w') as file:
        json.dump(template, file, indent=4, sort_keys=True)
    print('New template JSON file in: {}'.format(filename))

def _parse_json_data(json_dir):
    with open(json_dir, 'r') as file:
        args = json.load(file)
    return args


class TFNetworkParser(nnBase):
    '''
    The class takes a JSON file directory as input and construct an neural network with tensorflow
    Notes:
         The elements in holders do not necessarily be tf.placeholder
         The interface is for the convenience of taking different types of inputs,
         as long as they are tensorflow types, e.g. tf.DataSet.iterator.get_next_batch

         The class is only for training. See TFNetworkParserTest for testing stage.
    '''
    def __init__(self, holders, json_dir, model_dir=default_dir, pre_train=True):
        self.model_dir = model_dir
        self.json_data = _parse_json_data(json_dir)

        # If the parameter "use_softmax" is False in JSON, self.y_prob will be None
        self.y_prob, self.y_logits = self.parse_json(holders)
        self.loss = self.objective(holders['ys'])
        self.saver = tf.train.Saver()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.InteractiveSession(config=config)
        self.sess.run(tf.global_variables_initializer())

        if pre_train and len(os.listdir(self.model_dir)) != 0:
            _, self.counter = self.load()
        else:
            print('Build model from scratch!!')
            self.counter = 0

    def parse_json(self, holders):
        model = holders['xs']
        json_pairs = list(self.json_data['layers'].items())
        json_pairs.sort(key=lambda x: int(x[0].split('_')[-1]))

        with tf.variable_scope(self.json_data['model_name']):
            for name, param in json_pairs:
                model = self.build_from_args(model, name, param)

                # By default, do not apply batch_norm to the fully connected layers
                # So only convolution layer is considered here
                if self.json_data['use_batch_norm'] and 'conv' in name:
                    args = [model, name + '_bn', self.json_data['is_training']]
                    model = self.batch_norm(*args)

                # Activation function
                if self.json_data['activation'].lower() == 'relu':
                    model = tf.nn.relu(model)
                elif self.json_data['activation'].lower() == 'lrelu':
                    model = self.lrelu(model)

                # Similarly, only apply dropout to fully connected layers
                if 'dense' in name and self.json_data['use_dropout']:
                    model = tf.nn.dropout(model, keep_prob=self.json_data['keep_prob'])
            # The last layer is processed separately
            shape = model.get_shape().as_list()
            if len(shape) != 2:
                model = tf.reshape(model, [shape[0], -1], name='reshape_out')
            output = self.fully_connect(model, self.json_data['num_classes'], name='output')
            if self.json_data['use_softmax']:
                prob = tf.nn.softmax(output, name='softmax')
            else:
                prob = None
            return prob, output

    def build_from_args(self, model, name, param):
        if 'conv' in name:
            args = [model, param, name, self.json_data['conv_kernel_size'], self.json_data['conv_kernel_size'],
                    self.json_data['conv_stride'], self.json_data['conv_stride'],
                    not self.json_data['use_batch_norm']]
            return self.conv2d(*args)
        elif 'pool' in name:
            args = [model, name, 2, 2, self.json_data['pooling_stride'],
                    self.json_data['pooling_stride'], self.json_data['pooling_type']]
            return self.pooling(*args)
        elif 'dense' in name:
            shape = model.get_shape().as_list()
            if len(shape) != 2:
                model = tf.reshape(model, [shape[0], -1], name='reshaped')
            return self.fully_connect(model, param, name)
        else:
            raise AttributeError('Invalid input arguments')

    def objective(self, ys):
        if self.json_data['loss'] == 'cross_entropy':
            if self.json_data['sparse_ys']:
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=ys,
                                                                      logits=self.y_logits)
                return tf.reduce_mean(loss)
            else:
                loss = tf.nn.softmax_cross_entropy_with_logits(labels=ys,
                                                               logits=self.y_logits)
                return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))
        elif self.json_data['loss'] == 'least_square':
            loss = tf.square(self.y_logits - ys)
            return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))
        else:
            raise AttributeError('Invalid attribute [loss] in JSON')

    def __del__(self):
        self.sess.close()


class TFNetworkParserTest(TFNetworkParser):
    '''
    This class is the TFNetworkParserTest for testing stage
    '''
    def __init__(self, holders, json_dir, model_dir=default_dir, pre_train=True):
        super(TFNetworkParserTest, self).__init__(holders, json_dir, model_dir, pre_train)

    def parse_json(self, holders):
        model = holders['xs']
        json_pairs = list(self.json_data['layers'].items())
        json_pairs.sort(key=lambda x: int(x[0].split('_')[-1]))

        with tf.variable_scope(self.json_data['model_name']):
            for name, param in json_pairs:
                model = self.build_from_args(model, name, param)

                if self.json_data['use_batch_norm'] and 'conv' in name:
                    args = [model, name + '_bn', False]
                    model = self.batch_norm(*args)

                if self.json_data['activation'].lower() == 'relu':
                    model = tf.nn.relu(model)
                elif self.json_data['activation'].lower() == 'lrelu':
                    model = self.lrelu(model)

                # No dropout for testing network (set keep_prob=1.0)
                if 'dense' in name and self.json_data['use_dropout']:
                    model = tf.nn.dropout(model, keep_prob=1.0)
            # The last layer is processed separately
            shape = model.get_shape().as_list()
            if len(shape) != 2:
                model = tf.reshape(model, [shape[0], -1], name='reshape_out')
            output = self.fully_connect(model, self.json_data['num_classes'], name='output')
            if self.json_data['use_softmax']:
                prob = tf.nn.softmax(output, name='softmax')
            else:
                prob = None
            return prob, output


if __name__ == '__main__':
    # create_json_template('json2tf.json')
    xs = tf.placeholder(tf.float32, [8, 256, 256, 3], name='xs')
    ys = tf.placeholder(tf.int32, [8], name='ys')
    placeholders = {'xs': xs, 'ys': ys}

    model = TFNetworkParserTest(placeholders, 'json2tf.json')