from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
import keras
from keras.regularizers import l1_l2
from keras.layers import (Dense, Embedding, Input, LSTM, TimeDistributed, Lambda, Reshape, InputSpec, Wrapper, RNN, Dropout,
                          Multiply)
from keras_applications.resnet_common import ResNet101
from keras.optimizers import Adam
from models.word_vectors import get_word_vector_class
from keras.models import Model
import tensorflow as tf
from innvestigate.analyzer.relevance_based.relevance_analyzer import *
from innvestigate.examples.utils_imagenet import *
from scipy.special import expit as sigmoid
from scipy.special import softmax
from nltk.corpus import stopwords
import keras.backend as K


STOP_WORDS = list(set(stopwords.words('english')))
EPS = 0.01
ALPHA = 1
BETA = 0


# the base structure of the image captioning models
class ImgCaptioningAttentionModel(object):

    def __init__(self, config):
        self._learning_rate = config.learning_rate
        self._embedding_dim = config.embedding_dim
        self._hidden_dim = config.hidden_dim
        self._dropout_rate = config.drop_rate
        self._l1_reg = config.l1_reg
        self._l2_reg = config.l2_reg
        self._initializer = 'glorot_uniform'
        self._rnn_type = config.rnn_type
        self._rnn_layers = config.rnn_layers
        self._word_embedding_weights = config.pretrained_word_vector
        self._regularizer = l1_l2(self._l1_reg, self._l2_reg)
        self._bidirectional_rnn = config.bidirectional_rnn
        self._keras_model = None
        self.T = config.sentence_length
        self.L = config.img_feature_length
        self.D = config.img_feature_dim
        self.img_encoder = config.img_encoder
        self.layer_name = config.layer_name
        self.batch_size = config.batch_size

    def _build_image_embedding(self):
        if self.img_encoder == 'vgg16':
            base_model = VGG16(include_top=False, weights='imagenet')
        elif self.img_encoder == 'vgg19':
            base_model = VGG19(include_top=False, weights='imagenet')
        elif self.img_encoder == 'inception_v3':
            base_model = InceptionV3(include_top=False, weights='imagenet')
        else:
            raise ValueError("not implemented encoder type")

        self.image_model = Model(inputs=base_model.input, outputs=base_model.get_layer(self.layer_name).output)

        for layer in self.image_model.layers:
            layer.trainable = False

        image_embedding = Reshape((self.L, self.D))(self.image_model.output)
        return self.image_model.input, image_embedding

    def _build_word_embedding(self, vocabs, vocab_size):
        self._vocab_size = vocab_size
        sentence_input = Input(shape=(self.T,), name="captions_input")
        if self._word_embedding_weights is None:
            word_embedding = Embedding(
                input_dim=self._vocab_size,
                output_dim=self._embedding_dim,
                embeddings_regularizer=self._regularizer
            )(sentence_input)
        else:
            WordVector = get_word_vector_class(self._word_embedding_weights)
            word_vector = WordVector(vocabs, self._initializer)
            embedding_weights = word_vector.vectorize_words(vocabs)
            word_embedding = Embedding(
                input_dim=self._vocab_size,
                output_dim=self._embedding_dim,
                embeddings_regularizer=self._regularizer,
                weights=[embedding_weights]
            )(sentence_input)
        return sentence_input, word_embedding

    def build(self, vocabs, vocab_size):
        '''This function build up the image captioning models'''
        raise NotImplementedError()

    def categorical_crossentropy_from_logits(self, y_true, y_pred):
        y_true = y_true[:, :-1, :]  # Discard the last timestep/word (dummy)
        y_pred = y_pred[:, :-1, :]  # Discard the last timestep/word (dummy)

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_true, axis=-1),
                                                                             logits=y_pred))
        return loss

    def categorical_accuracy_with_variable_timestep(self, y_true, y_pred):
        y_true = y_true[:, :-1, :]  # Discard the last timestep/word (dummy)
        y_pred = y_pred[:, :-1, :]  # Discard the last timestep/word (dummy)

        # Flatten the timestep dimension
        shape = tf.shape(y_true)
        y_true = tf.reshape(y_true, [-1, shape[-1]])
        y_pred = tf.reshape(y_pred, [-1, shape[-1]])

        # Discard rows that are all zeros as they represent dummy or padding words.
        is_zero_y_true = tf.equal(y_true, 0)
        is_zero_row_y_true = tf.reduce_all(is_zero_y_true, axis=-1)
        y_true = tf.boolean_mask(y_true, ~is_zero_row_y_true)
        y_pred = tf.boolean_mask(y_pred, ~is_zero_row_y_true)

        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_true, axis=1),
                                                   tf.argmax(y_pred, axis=1)),
                                          dtype=tf.float32))
        return accuracy

    @property
    def keras_model(self):
        return self.training_model
# Our code is based the attented augmented LSTM https://gist.github.com/wassname/5292f95000e409e239b9dc973295327a
class ExternalAttentionRNNWrapper(Wrapper):
    """
        The basic idea of the implementation is based on the paper:
            "Effective Approaches to Attention-based Neural Machine Translation" by Luong et al.

        This layer is an attention layer, which can be wrapped around arbitrary RNN layers.
        This way, after each time step an attention vector is calculated
        based on the current output of the LSTM and the entire input time series.
        This attention vector is then used as a weight vector to choose special values
        from the input data. This data is then finally concatenated to the next input
        time step's data. On this a linear transformation in the same space as the input data's space
        is performed before the data is fed into the RNN cell again.

        This technique is similar to the input-feeding method described in the paper cited.

        The only difference compared to the AttentionRNNWrapper is, that this layer
        applies the attention layer not on the time-depending input but on a second
        time-independent input (like image clues) as described in:
            Show, Attend and Tell: Neural Image Caption Generation with Visual Attention
            https://arxiv.org/abs/1502.03044
    """

    def __init__(self, layer, weight_initializer="glorot_uniform", return_attention=False, **kwargs):
        assert isinstance(layer, RNN)
        self.layer = layer
        self.supports_masking = True
        self.weight_initializer = weight_initializer
        self.bias_initializer = 'zeros'
        self.return_attention = return_attention
        self._num_constants = None
        # self.hidden_size = hidden_size

        super(ExternalAttentionRNNWrapper, self).__init__(layer, **kwargs)

        self.input_spec = [InputSpec(ndim=3), InputSpec(ndim=3)]  # InputSpec stores the ndim, dtype and shape of input

    def _validate_input_shape(self, input_shape):
        if len(input_shape) >= 2:
            if len(input_shape[:2]) != 2:
                raise ValueError(
                    "Layer has to receive two inputs: the temporal signal and the external signal which is constant for all time steps")
            if len(input_shape[0]) != 3:
                raise ValueError(
                    "Layer received a temporal input with shape {0} but expected a Tensor of rank 3.".format(
                        input_shape[0]))
            if len(input_shape[1]) != 3:
                raise ValueError(
                    "Layer received a time-independent input with shape {0} but expected a Tensor of rank 3.".format(
                        input_shape[1]))
        else:
            raise ValueError(
                "Layer has to receive at least 2 inputs: the temporal signal and the external signal which is constant for all time steps")

    def build(self, input_shape):
        self._validate_input_shape(input_shape)
        super(ExternalAttentionRNNWrapper, self).build(input_shape)
        # print(input_shape)
        embed_dim = input_shape[0][-1]  # embed_dim
        static_input_dim = input_shape[1][-1]  # input feature sequence dim of a single feature D
        for i, x in enumerate(input_shape):
            self.input_spec[i] = InputSpec(shape=x)

        if not self.layer.built:
            self.layer.build((None, None, embed_dim + embed_dim))
            self.layer.built = True

        if self.layer.return_sequences:
            output_dim = self.layer.compute_output_shape(input_shape[0])[0][-1]
        else:
            output_dim = self.layer.compute_output_shape(input_shape[0])[-1]
        temporal_input_dim = output_dim
        self._W1 = self.add_weight(shape=(static_input_dim, temporal_input_dim), name="{}_W1".format(self.name),
                                   initializer=self.weight_initializer)
        self._W2 = self.add_weight(shape=(output_dim, temporal_input_dim), name="{}_W2".format(self.name),
                                   initializer=self.weight_initializer)
        self._b1 = self.add_weight(shape=(temporal_input_dim,), name="{}_b2".format(self.name),
                                   initializer=self.weight_initializer)
        self._V = self.add_weight(shape=(temporal_input_dim, 1), name="{}_V".format(self.name),
                                  initializer=self.weight_initializer)

    @property
    def trainable_weights(self):
        return self._trainable_weights + self.layer.trainable_weights

    @property
    def non_trainable_weights(self):
        return self._non_trainable_weights + self.layer.non_trainable_weights

    def compute_output_shape(self, input_shape):
        self._validate_input_shape(input_shape)

        output_dim = self.layer.compute_output_shape(input_shape[0])[0][-1]
        output_shape = (None, None, output_dim)

        if self.layer.return_state:
            if not isinstance(output_shape, list):
                output_shape = [output_shape]
            output_shape = output_shape + [(None, output_dim)] + [(None, output_dim)]

        if self.return_attention:
            if not isinstance(output_shape, list):
                output_shape = [output_shape]

            output_shape = output_shape + [(None, input_shape[1][1])]  # hid_dim, L(num_features)

        return output_shape

    def step(self, x, states):
        global_img_feature = states[2]
        input_x = K.concatenate([x, global_img_feature], axis=-1)
        h, new_states = self.layer.cell.call(input_x, states[:2])
        X_static = states[3]
        total_x_static_prod = states[4]
        hw = K.expand_dims(K.dot(h, self._W2), 1)
        additive_atn = total_x_static_prod + hw
        additive_atn = K.tanh(additive_atn)
        attention = K.softmax(K.dot(additive_atn, self._V), axis=1)
        static_x_weighted = K.sum(attention * X_static, [1])
        attention = K.squeeze(attention, -1)
        h = K.concatenate([h + static_x_weighted, attention])
        return h, new_states

    def call(self, x, constants=None, mask=None, initial_state=None):
        input_shape = self.input_spec[0].shape
        if len(x) > 2:
            initial_state = x[2:4]
            x = x[:2]
            assert len(initial_state) >= 1

        self.static_x = x[1]

        x = x[0]

        if self.layer.stateful:
            initial_states = self.layer.states
        elif initial_state is not None:
            initial_states = initial_state
            if not isinstance(initial_states, (list, tuple)):
                initial_states = [initial_states]
        else:
            initial_states = self.layer.get_initial_state(x)
        if constants is None:
            constants = []
        constants += self.get_constants(self.static_x)
        last_output, outputs, states = K.rnn(
            self.step,
            x,
            initial_states,
            go_backwards=self.layer.go_backwards,
            mask=mask,
            constants=constants,
            unroll=self.layer.unroll,
            input_length=input_shape[1]
        )

        output_dim = self.layer.compute_output_shape(input_shape)[0][-1]

        last_output = last_output[:output_dim]

        attentions = outputs[:, :, output_dim:]
        outputs = outputs[:, :, :output_dim]

        if self.layer.stateful:
            self.updates = []
            for i in range(len(states)):
                self.updates.append((self.layer.states[i], states[i]))

        if self.layer.return_sequences:
            output = outputs
        else:
            output = last_output

        if getattr(last_output, '_uses_learning_phase', False):
            output._uses_learning_phase = True
            for state in states:
                state._uses_learning_phase = True

        if self.layer.return_state:
            if not isinstance(states, (list, tuple)):
                states = [states]
            else:
                states = list(states)
            output = [output] + states

        if self.return_attention:
            if not isinstance(output, list):
                output = [output]
            output = output + [attentions]
        return output

    def _standardize_args(self, inputs, initial_state, constants, num_constants):
        """Standardize `__call__` to a single list of tensor inputs.

        When running a model loaded from file, the input tensors
        `initial_state` and `constants` can be passed to `RNN.__call__` as part
        of `inputs` instead of by the dedicated keyword arguments. This method
        makes sure the arguments are separated and that `initial_state` and
        `constants` are lists of tensors (or None).

        # Arguments
        inputs: tensor or list/tuple of tensors
        initial_state: tensor or list of tensors or None
        constants: tensor or list of tensors or None

        # Returns
        inputs: tensor
        initial_state: list of tensors or None
        constants: list of tensors or None
        """
        if isinstance(inputs, list) and len(inputs) > 2:
            assert initial_state is None and constants is None
            if num_constants is not None:
                constants = inputs[-num_constants:]
                inputs = inputs[:-num_constants]
            initial_state = inputs[2:]
            inputs = inputs[:2]

        def to_list_or_none(x):
            if x is None or isinstance(x, list):
                return x
            if isinstance(x, tuple):
                return list(x)
            return [x]

        initial_state = to_list_or_none(initial_state)
        constants = to_list_or_none(constants)

        return inputs, initial_state, constants

    def __call__(self, inputs, initial_state=None, constants=None, **kwargs):

        inputs, initial_state, constants = self._standardize_args(
            inputs, initial_state, constants, self._num_constants)


        if initial_state is None and constants is None:
            return super(ExternalAttentionRNNWrapper, self).__call__(inputs, **kwargs)


        additional_inputs = []
        additional_specs = []
        if initial_state is not None:
            kwargs['initial_state'] = initial_state
            additional_inputs += initial_state
            self.state_spec = [InputSpec(shape=K.int_shape(state))
                               for state in initial_state]
            additional_specs += self.state_spec
        if constants is not None:
            kwargs['constants'] = constants
            additional_inputs += constants
            self.constants_spec = [InputSpec(shape=K.int_shape(constant))
                                   for constant in constants]
            self._num_constants = len(constants)
            additional_specs += self.constants_spec
        is_keras_tensor = K.is_keras_tensor(additional_inputs[0])
        for tensor in additional_inputs:
            if K.is_keras_tensor(tensor) != is_keras_tensor:
                raise ValueError('The initial state or constants of an ExternalAttentionRNNWrapper'
                                 ' layer cannot be specified with a mix of'
                                 ' Keras tensors and non-Keras tensors'
                                 ' (a "Keras tensor" is a tensor that was'
                                 ' returned by a Keras layer, or by `Input`)')

        if is_keras_tensor:
            full_input = inputs + additional_inputs
            full_input_spec = self.input_spec + additional_specs
            original_input_spec = self.input_spec
            self.input_spec = full_input_spec
            output = super(ExternalAttentionRNNWrapper, self).__call__(full_input, **kwargs)
            self.input_spec = self.input_spec[:len(original_input_spec)]
            return output
        else:
            return super(ExternalAttentionRNNWrapper, self).__call__(inputs, **kwargs)

    def get_constants(self, x):
        constants = [x, K.dot(x, self._W1) + self._b1]
        return constants

    def get_config(self):
        config = {'return_attention': self.return_attention, 'weight_initializer': self.weight_initializer}
        base_config = super(ExternalAttentionRNNWrapper, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



# the adaptive attention captioning model
class ImgCaptioningAdaptiveAttentionModel(ImgCaptioningAttentionModel):


    def _build_image_embedding(self):
        if self.img_encoder == 'vgg16':
            base_model = VGG16(include_top=False, weights='imagenet')
        elif self.img_encoder == 'vgg19':
            base_model = VGG19(include_top=False, weights='imagenet')
        elif self.img_encoder == 'inception_v3':
            base_model = InceptionV3(include_top=False, weights='imagenet')
        elif self.img_encoder == 'resnet101':
            base_model = ResNet101(include_top=False, weights='imagenet', backend=keras.backend, layers=keras.layers,
                                   models=keras.models, utils=keras.utils)
        else:
            raise ValueError("not implemented encoder type")
        # VGG16(include_top=False, weights='imagenet', pooling='avg').summary()

        self.image_model = Model(inputs=base_model.input, outputs=base_model.get_layer(self.layer_name).output)

        for layer in self.image_model.layers:
            layer.trainable = True
        '''with batchnormlization'''
        image_embedding = Reshape((self.L, self.D))(self.image_model.output)
        return self.image_model.input, image_embedding

    def build(self, vocabs, vocab_size):

        self.image_input, self.image_features_input = self._build_image_embedding()

        self.averaged_image_features = Lambda(lambda x: K.mean(x, axis=1))
        self.averaged_image_features = self.averaged_image_features(self.image_features_input)

        self.image_features = TimeDistributed(Dense(self._hidden_dim, activation='relu'), name='image_features')(
            self.image_features_input)
        self.image_features = Dropout(rate=self._dropout_rate)(self.image_features)

        self.global_image_feature = Dense(self._embedding_dim, activation='relu', name='global_img_feature')(
            self.averaged_image_features)  # dense 3
        self.global_image_feature = Dropout(rate=self._dropout_rate)(self.global_image_feature)

        self.captions_input, self.captions = self._build_word_embedding(vocabs, vocab_size)

        self.decoder = LSTM(self._hidden_dim, return_sequences=True, return_state=True,recurrent_activation='sigmoid',
                            recurrent_dropout=self._dropout_rate, dropout=self._dropout_rate)
        self.attented_decoder = ExternalAttentionRNNWrapperLocalAttentionV3(self.decoder, return_attention=True)
        self.output = TimeDistributed(Dense(self._vocab_size), name="output")

        self.attented_encoder_training_data, c, a, b = self.attented_decoder([self.captions, self.image_features],
                                                                             constants=[self.global_image_feature])
        self.attented_encoder_training_data = Dropout(rate=self._dropout_rate)(self.attented_encoder_training_data)
        self.training_output_data = self.output(self.attented_encoder_training_data)
        self.training_output_data = Dropout(rate=self._dropout_rate)(self.training_output_data)

        self.training_model = Model(inputs=[self.captions_input, self.image_input], outputs=self.training_output_data)

        self.training_model.compile(optimizer=Adam(lr=self._learning_rate, clipvalue=0.1),
                                    loss=self.categorical_crossentropy_from_logits,
                                    metrics=[self.categorical_accuracy_with_variable_timestep])
# this is the adaptive attention augmented LSTM decoder
class ExternalAttentionRNNWrapperLocalAttentionV3(ExternalAttentionRNNWrapper):

    def call(self, x, constants=None, mask=None, initial_state=None):
        input_shape = self.input_spec[0].shape
        if len(x) > 2:
            x = x[:2]
        self.static_x = x[1]
        x = x[0]
        if self.layer.stateful:
            initial_states = self.layer.states
        elif initial_state is not None:
            initial_states = initial_state
            if not isinstance(initial_states, (list, tuple)):
                initial_states = [initial_states]
        else:
            initial_states = self.layer.get_initial_state(x)
        if constants is None:
            constants = []
        constants += self.get_constants(self.static_x)

        last_output, outputs, states = K.rnn(
            self.step,
            x,
            initial_states,
            go_backwards=self.layer.go_backwards,
            mask=mask,
            constants=constants,
            unroll=self.layer.unroll,
            input_length=input_shape[1]
        )
        output_dim = self.layer.compute_output_shape(input_shape)[0][-1]

        last_output = last_output[:output_dim]

        attentions = outputs[:, :, output_dim:]
        outputs = outputs[:, :, :output_dim]

        if self.layer.stateful:
            self.updates = []
            for i in range(len(states)):
                self.updates.append((self.layer.states[i], states[i]))

        if self.layer.return_sequences:
            output = outputs
        else:
            output = last_output
        if getattr(last_output, '_uses_learning_phase', False):
            output._uses_learning_phase = True
            for state in states:
                state._uses_learning_phase = True
        if self.layer.return_state:
            if not isinstance(states, (list, tuple)):
                states = [states]
            else:
                states = list(states)
            output = [output] + states

        if self.return_attention:
            if not isinstance(output, list):
                output = [output]
            output = output + [attentions]
        return output

    def build(self, input_shape):
        self._validate_input_shape(input_shape)
        embed_dim = input_shape[0][-1]  # embed_dim
        static_input_dim = input_shape[1][-1]  # input feature sequence dim of a single feature D

        for i, x in enumerate(input_shape):
            self.input_spec[i] = InputSpec(shape=x)

        if not self.layer.built:
            self.layer.build((None, None, embed_dim * 2))
            self.layer.built = True

        if self.layer.return_sequences:
            output_dim = self.layer.compute_output_shape(input_shape[0])[0][-1]
        else:
            output_dim = self.layer.compute_output_shape(input_shape[0])[-1]

        temporal_input_dim = output_dim
        self._Wv = self.add_weight(shape=(static_input_dim, temporal_input_dim), name="{}_Wv".format(self.name),
                                   initializer=self.weight_initializer)

        self._Wg = self.add_weight(shape=(output_dim, temporal_input_dim), name="{}_Wg".format(self.name),
                                   initializer=self.weight_initializer)

        self._Wx = self.add_weight(shape=(embed_dim * 2, output_dim), name="{}_Wx".format(self.name),
                                  initializer=self.weight_initializer)

        self._Wh = self.add_weight(shape=(output_dim, output_dim), name="{}_Wh".format(self.name),
                                  initializer=self.weight_initializer)

        self._Ws = self.add_weight(shape=(output_dim, temporal_input_dim), name="{}_Ws".format(self.name),
                                  initializer=self.weight_initializer)

        self._V = self.add_weight(shape=(temporal_input_dim, 1), name="{}_V".format(self.name),
                                  initializer=self.weight_initializer)

    def step(self, x, states):
        # states[0]: h
        # states[1]: c
        # states[2]: global_img_feature
        # states[3]:static_x,
        # states[4]: total_x_static_proj
        htm1 = states[0]
        global_img_feature = states[2]
        input_x = K.concatenate([x, global_img_feature], axis=-1)
        h, new_states = self.layer.cell.call(input_x, states[:2])
        ct = new_states[1]
        st = K.tanh(ct) * K.sigmoid(K.dot(input_x, self._Wx) + K.dot(htm1, self._Wh))
        st = K.expand_dims(st, 1)
        zt_extend = K.dot((K.tanh(K.dot(st, self._Ws) + K.expand_dims(K.dot(h, self._Wg), 1))), self._V)
        X_static = states[3] # (L, D)
        total_x_static_prod = states[4]  # (L, output_dim)
        hw = K.expand_dims(K.dot(h, self._Wg), 1)  # (1, output_dim)
        additive_atn = total_x_static_prod + hw  # (L, output_dim)
        additive_atn = K.tanh(additive_atn)
        attention = K.softmax(K.dot(additive_atn, self._V), axis=1)  # (L, 1)
        alpha_hat = K.concatenate([K.dot(additive_atn, self._V), zt_extend], axis=1)
        alpha_hat = K.softmax(alpha_hat, axis=1)
        beta = alpha_hat[:, -1, :]
        static_x_weighted = K.sum(attention * X_static, [1])  # (1, static_input_dim)
        context_hat = beta * K.squeeze(st, axis=1) + (1-beta) * static_x_weighted
        attention = K.squeeze(attention, -1)
        h = K.concatenate([h+context_hat, attention])
        return h, new_states

    def get_constants(self, x):
        constants = [x, K.dot(x, self._Wv)]
        return constants



# the gridTD attention model, with two LSTM layers and the adaptive module
class ImgCaptioninggridTDAdaptiveModel(ImgCaptioningAttentionModel):

    def _build_image_embedding(self):
        if self.img_encoder == 'vgg16':
            base_model = VGG16(include_top=False, weights='imagenet')
        elif self.img_encoder == 'vgg19':
            base_model = VGG19(include_top=False, weights='imagenet')
        elif self.img_encoder == 'inception_v3':
            base_model = InceptionV3(include_top=False, weights='imagenet')
        elif self.img_encoder == 'resnet101':
            base_model = ResNet101(include_top=False, weights='imagenet', backend=keras.backend, layers=keras.layers,
                                   models=keras.models, utils=keras.utils)
        else:
            raise ValueError("not implemented encoder type")
        self.image_model = Model(inputs=base_model.input, outputs=base_model.get_layer(self.layer_name).output)

        for layer in self.image_model.layers:
            layer.trainable = True
        image_embedding = Reshape((self.L, self.D))(self.image_model.output)
        return self.image_model.input, image_embedding

    def build(self, vocabs, vocab_size):
        self.image_input, self.image_features_input = self._build_image_embedding()
        self.image_features = TimeDistributed(Dense(self._hidden_dim, activation='relu'), name='image_features')(
            self.image_features_input)
        self.image_features = Dropout(rate=self._dropout_rate)(self.image_features)
        self.captions_input, self.captions = self._build_word_embedding(vocabs, vocab_size)

        self.averaged_image_features = Lambda(lambda x: K.mean(x, axis=1))
        self.averaged_image_features = self.averaged_image_features(self.image_features_input)

        self.global_image_feature = Dense(self._embedding_dim, activation='relu', name='global_img_feature')(
            self.averaged_image_features)  # dense 3
        self.global_image_feature = Dropout(rate=self._dropout_rate)(self.global_image_feature)

        self.language_lstm = LSTM(self._hidden_dim, return_sequences=True, return_state=True, recurrent_activation='sigmoid',
                                  recurrent_dropout=self._dropout_rate, dropout=self._dropout_rate)

        self.attented_encoder = ExternalBottomUpAttentionAdaptive(self.language_lstm, return_attention=True)

        self.output = TimeDistributed(Dense(self._vocab_size), name="output")

        self.attented_encoder_training_data, h, c, self.attention = self.attented_encoder(
            [self.captions, self.image_features], constants=[self.global_image_feature])

        self.attented_encoder_training_data = Dropout(rate=self._dropout_rate)(self.attented_encoder_training_data)

        self.training_output_data = self.output(self.attented_encoder_training_data)

        self.training_output_data = Dropout(rate=self._dropout_rate)(self.training_output_data)

        self.training_model = Model(inputs=[self.captions_input, self.image_input], outputs=self.training_output_data)

        self.training_model.compile(optimizer=Adam(lr=self._learning_rate, clipvalue=0.1, beta_1=0.8, beta_2=0.999),
                                    loss='categorical_crossentropy',
                                    metrics=[self.categorical_accuracy_with_variable_timestep])
# This is the gridTD attention augmented LSTM decoder
class ExternalBottomUpAttentionAdaptive(ExternalAttentionRNNWrapper):

    def top_down_lstm_forward(self, x, h, c, weight_i, weight_h, bias):
        z = K.dot(x, weight_i)
        z += K.dot(h, weight_h)
        z = z + bias
        z0 = z[:, :self.hidden_dim]
        z1 = z[:, self.hidden_dim: 2 * self.hidden_dim]
        z2 = z[:, 2 * self.hidden_dim: 3 * self.hidden_dim]
        z3 = z[:, 3 * self.hidden_dim:]
        i = K.sigmoid(z0)
        f = K.sigmoid(z1)
        c = f * c + i * K.tanh(z2)
        o = K.sigmoid(z3)
        ht = o * K.tanh(c)
        ct = c
        return ht, [ht, ct]

    def build(self, input_shape):
        self._validate_input_shape(input_shape)
        super(ExternalAttentionRNNWrapper, self).build()
        for i, x in enumerate(input_shape):
            self.input_spec[i] = InputSpec(shape=x)
        if self.layer.return_sequences:
            self.hidden_dim = self.layer.compute_output_shape(input_shape[0])[0][-1]
        else:
            self.hidden_dim = self.layer.compute_output_shape(input_shape[0])[-1]

        self.feature_dim = input_shape[1][-1]
        self.embedding_dim = input_shape[0][-1]
        self.hidden_att_dim = input_shape[1][-1]

        if not self.layer.built:

            self.layer.build((None, None, self.hidden_dim * 2))
            self.layer.built = True
        self._W_va = self.add_weight(shape=(self.feature_dim, self.hidden_att_dim), name="{}_W_va".format(self.name),
                                     initializer=self.weight_initializer)
        self._W_ha = self.add_weight(shape=(self.hidden_dim, self.hidden_att_dim), name='{}_W_ha'.format(self.name),
                                     initializer=self.weight_initializer)
        self._W_a = self.add_weight(shape=(self.hidden_att_dim, 1), name='{}_W_a'.format(self.name),
                                    initializer=self.weight_initializer)

        self._W_x = self.add_weight(shape=(self.hidden_dim + self.embedding_dim * 2, self.hidden_dim),
                                    name="{}_W_x".format(self.name),
                                    initializer=self.weight_initializer)
        self._W_h = self.add_weight(shape=(self.hidden_dim, self.hidden_dim), name="{}_W_h".format(self.name),
                                    initializer=self.weight_initializer)
        self._W_s = self.add_weight(shape=(self.hidden_dim, self.hidden_att_dim), name="{}_W_s".format(self.name),
                                    initializer=self.weight_initializer)

        self._top_down_lstm_weight_i = self.add_weight(
            shape=(self.embedding_dim + self.feature_dim + self.hidden_dim, 4 * self.hidden_dim),
            name='{}_top_down_lstm_weight_i'.format(self.name), initializer=self.weight_initializer)
        self._top_down_lstm_weight_h = self.add_weight(shape=(self.hidden_dim, 4 * self.hidden_dim),
                                                       name='{}_top_down_lstm_weight_h'.format(self.name),
                                                       initializer='orthogonal')
        self._top_down_lstm_weight_bias = self.add_weight(shape=(4 * self.hidden_dim,),
                                                          name='{}_top_down_lstm_weight_bias'.format(self.name),
                                                          initializer='zeros')

    def call(self, x, constants=None, mask=None, initial_state=None):

        input_shape = self.input_spec[0].shape

        if len(x) > 2:
            x = x[:2]

        static_x = x[1]
        x = x[0]
        if initial_state is not None:
            initial_states = initial_state
            if not isinstance(initial_states, (list, tuple)):
                initial_states = [initial_states]
        else:
            initial_states = self.layer.get_initial_state(x) + self.layer.get_initial_state(x)
        if not constants:
            constants = []
        constants += self.get_constants(static_x)
        last_output, outputs, states = K.rnn(
            self.step,
            x,
            initial_states,  # h1 c1 h2 c2
            go_backwards=False,
            mask=mask,
            constants=constants,
            unroll=False,
            input_length=input_shape[1]
        )
        output_dim = self.hidden_dim
        last_output = last_output[:, : output_dim]
        attentions = outputs[:, :, output_dim + self.feature_dim:]
        outputs = outputs[:, :, :output_dim]

        if self.layer.return_sequences:
            output = outputs
        else:
            output = last_output

        if getattr(last_output, '_uses_learning_phase', False):
            output._uses_learning_phase = True
            for state in states:
                state._uses_learning_phase = True

        if self.layer.return_state:
            if not isinstance(states, (list, tuple)):
                states = [states]
            else:
                states = list(states)
            output = [output] + states[2:4]

        if self.return_attention:
            if not isinstance(output, list):
                output = [output]
            output = output + [attentions]

        return output

    def step(self, x, states):
        h1 = states[0] # output_dim,
        c1 = states[1] # output_dim,
        h2 = states[2] # output_dim,
        c2 = states[3] # output_dim,
        static_x_mean = states[4] # emb_dim,
        static_x = states[5] # L, output_dim,
        static_x_project = states[6]  # L, output_dim
        top_down_lstm_input = K.concatenate((h2, static_x_mean, x))
        h1_new, h1_states_new = self.top_down_lstm_forward(top_down_lstm_input, h1, c1, self._top_down_lstm_weight_i,
                                                           self._top_down_lstm_weight_h,
                                                           self._top_down_lstm_weight_bias)
        c1_new = h1_states_new[1]
        st = K.tanh(c1_new) * K.sigmoid(K.dot(top_down_lstm_input, self._W_x) + K.dot(h1, self._W_h))
        st = K.expand_dims(st, 1)
        zt_extend = K.dot((K.tanh(K.dot(st, self._W_s) + K.expand_dims(K.dot(h1_new, self._W_ha), 1))), self._W_a)
        h_proj = K.expand_dims(K.dot(h1_new, self._W_ha), 1)  # 1, output_dim
        attention_bf_act = K.dot(K.tanh(static_x_project + h_proj), self._W_a)
        attention = K.softmax(attention_bf_act, axis=1)  # L, 1

        alpha_hat = K.concatenate([attention_bf_act, zt_extend], axis=1)
        alpha_hat = K.softmax(alpha_hat, axis=1)
        beta = alpha_hat[:, -1, :]
        context = K.sum(attention * static_x, [1])  # 1, output_dim
        context_hat = beta * K.squeeze(st, axis=1) + (1 - beta) * context

        language_lstm_input = K.concatenate((context_hat, h1_new))

        h2_new, h2_states_new = self.layer.cell.call(language_lstm_input, [h2, c2])

        new_states = h1_states_new + h2_states_new

        return_h = K.concatenate([h2_new + context_hat, context, K.squeeze(attention, -1)])

        return return_h, new_states

    def get_constants(self, x):
        # add constants to speed up calculation
        x_project = K.dot(x, self._W_va)
        return [x, x_project]


# LRP inference model, adding the LRP weighted predicted scores, model weights are updated after training each batch
# The LRP weights are set as an Input tensor, the gradients are not backwards through the LRP algorithm
class ImgCaptioningGridTDLRPInferenceModel(ImgCaptioningAttentionModel):
    def __init__(self, config, dataset_provider):
        super(ImgCaptioningGridTDLRPInferenceModel, self).__init__(config)
        self.config = config
        self._dataset_provider = dataset_provider
    def _build_image_embedding(self):
        if self.img_encoder == 'vgg16':
            base_model = VGG16(include_top=False, weights='imagenet')
        elif self.img_encoder == 'vgg19':
            base_model = VGG19(include_top=False, weights='imagenet')
        elif self.img_encoder == 'inception_v3':
            base_model = InceptionV3(include_top=False, weights='imagenet')
        else:
            raise ValueError("not implemented encoder type")

        self.image_model = Model(inputs=base_model.input, outputs=base_model.get_layer(self.layer_name).output)

        for layer in self.image_model.layers:
            layer.trainable = True

        image_embedding = Reshape((self.L, self.D))(self.image_model.output)
        return self.image_model.input, image_embedding
    def build(self, vocabs, vocab_size):
        self.image_input, self.image_features_input = self._build_image_embedding()
        self.image_features = TimeDistributed(Dense(self._hidden_dim, activation='relu'), name='image_features')(
            self.image_features_input)  # time_distributed_1
        self.image_features = Dropout(rate=self._dropout_rate)(self.image_features)
        self.captions_input, self.captions = self._build_word_embedding(vocabs, vocab_size)

        self.averaged_image_features = Lambda(lambda x: K.mean(x, axis=1))
        self.averaged_image_features = self.averaged_image_features(self.image_features_input)

        self.global_image_feature = Dense(self._embedding_dim, activation='relu', name='global_img_feature')(
            self.averaged_image_features)  # dense 3
        self.global_image_feature = Dropout(rate=self._dropout_rate)(self.global_image_feature)

        self.language_lstm = LSTM(self._hidden_dim, return_sequences=True, return_state=True, recurrent_activation='sigmoid',
                                  recurrent_dropout=self._dropout_rate, dropout=self._dropout_rate)

        self.attented_encoder = ExternalBottomUpAttentionAdaptive(self.language_lstm, return_attention=True)

        self.output = TimeDistributed(Dense(self._vocab_size), name="output")

        self.attented_encoder_training_data, h, c, self.attention = self.attented_encoder(
            [self.captions, self.image_features], constants=[self.global_image_feature])

        self.attented_encoder_training_data = Dropout(rate=self._dropout_rate)(self.attented_encoder_training_data)

        self.training_output_data = self.output(self.attented_encoder_training_data)
        self.training_output_data = Dropout(rate=self._dropout_rate)(self.training_output_data)
        self.lrp_weight = Input(shape=self.training_output_data.shape[1:])
        self.training_output_data_lrp = Multiply()([self.training_output_data, self.lrp_weight])
        self.training_model = Model(inputs=[self.captions_input, self.image_input, self.lrp_weight],
                                    outputs=[self.training_output_data, self.training_output_data_lrp])

        self.training_model.compile(optimizer=Adam(lr=self._learning_rate, clipvalue=0.1),
                                    loss=[self.categorical_crossentropy_from_logits,
                                          self.categorical_crossentropy_from_logits],
                                    loss_weights=[0.5,0.5],
                                    metrics=[self.categorical_accuracy_with_variable_timestep])
class ImgCaptioningAdaptiveAttentionLRPInferenceModel(ImgCaptioningAttentionModel):
    def __init__(self, config, dataset_provider):
        super(ImgCaptioningAdaptiveAttentionLRPInferenceModel, self).__init__(config)
        self.config = config
        self._dataset_provider = dataset_provider

    def _build_image_embedding(self):

        if self.img_encoder == 'vgg16':
            base_model = VGG16(include_top=False, weights='imagenet')
        elif self.img_encoder == 'vgg19':
            base_model = VGG19(include_top=False, weights='imagenet')
        elif self.img_encoder == 'inception_v3':
            base_model = InceptionV3(include_top=False, weights='imagenet')
        else:
            raise ValueError("not implemented encoder type")

        self.image_model = Model(inputs=base_model.input, outputs=base_model.get_layer(self.layer_name).output)

        for layer in self.image_model.layers:
            layer.trainable = True

        image_embedding = Reshape((self.L, self.D))(self.image_model.output)
        return self.image_model.input, image_embedding

    def build(self, vocabs, vocab_size):
        self.image_input, self.image_features_input = self._build_image_embedding()

        self.averaged_image_features = Lambda(lambda x: K.mean(x, axis=1))
        self.averaged_image_features = self.averaged_image_features(self.image_features_input)

        self.image_features = TimeDistributed(Dense(self._hidden_dim, activation='relu'), name='image_features')(
            self.image_features_input)  # time_distributed_1
        self.image_features = Dropout(rate=self._dropout_rate)(self.image_features)

        self.global_image_feature = Dense(self._embedding_dim, activation='relu', name='global_img_feature')(
            self.averaged_image_features)  # dense 3
        self.global_image_feature = Dropout(rate=self._dropout_rate)(self.global_image_feature)

        self.captions_input, self.captions = self._build_word_embedding(vocabs, vocab_size)

        self.encoder = LSTM(self._hidden_dim, return_sequences=True, return_state=True,
                            recurrent_dropout=self._dropout_rate, activation='tanh',
                            recurrent_activation='sigmoid', dropout=self._dropout_rate)
        self.attented_encoder = ExternalAttentionRNNWrapperLocalAttentionV3(self.encoder, return_attention=True)
        self.output = TimeDistributed(Dense(self._vocab_size), name="output")

        self.attented_encoder_training_data, c, a, b = self.attented_encoder([self.captions, self.image_features],
                                                                             constants=[self.global_image_feature])
        self.attented_encoder_training_data = Dropout(rate=self._dropout_rate)(self.attented_encoder_training_data)
        self.training_output_data = self.output(self.attented_encoder_training_data)
        self.lrp_weight = Input(shape=self.training_output_data.shape[1:])
        self.training_output_data_lrp = Multiply()([self.training_output_data, self.lrp_weight])
        self.training_model = Model(inputs=[self.captions_input, self.image_input, self.lrp_weight],
                                    outputs=[self.training_output_data, self.training_output_data_lrp])

        self.training_model.compile(optimizer=Adam(lr=self._learning_rate, clipvalue=0.01),
                                    loss=[self.categorical_crossentropy_from_logits,
                                          self.categorical_crossentropy_from_logits],
                                    loss_weights=[0.5,0.5],
                                    metrics=[self.categorical_accuracy_with_variable_timestep])



#The external layer calculating the LRP heatmaps given the image input and the predicted captions, words in STOP_WORDS are excluded'''
class LRPInferenceLayerAdaptive(object):

    def __init__(self, model, dataset_provider, hidden_dim, embedding_dim, L, D, img_encoder, lrp_inference_mode):
        self._preprocessor = dataset_provider.caption_preprocessor
        self._EOS_ENCODED = self._preprocessor.EOS_TOKEN_LABEL_ENCODED
        self._SOS_ENCODER = self._preprocessor.SOS_TOKEN_LABEL_ENCODED
        self._dataset_provider = dataset_provider
        self._max_caption_length = 20
        self._hidden_dim = hidden_dim
        self._embedding_dim = embedding_dim
        self._lrp_inference_mode = lrp_inference_mode
        self.L = L
        self.D = D
        self._img_encoder = img_encoder
        if self._img_encoder in ['vgg16', 'vgg19']:
            self._color_conversion = 'BGRtoRGB'
            self._reshape_size = (14, 14)
            self._upscale = 16
        self._keras_model = model
        self._image_model = Model(inputs=self._keras_model.get_layer('input_1').input,
                                  outputs=self._keras_model.get_layer('block5_conv3').output)
        self._CNN_explainer = LRPSequentialPresetA(self._image_model, epsilon=EPS, neuron_selection_mode='replace')
        self._image_features_wieght, self._image_features_bias = self._keras_model.get_layer(
            'image_features').get_weights()
        self._global_img_feature_weight, self._global_img_feature_bias = self._keras_model.get_layer(
            'global_img_feature').get_weights()

        self._embedding = Model(self._keras_model.get_layer('embedding_1').input,
                                self._keras_model.get_layer('embedding_1').output)
        self._attention_layer = self._keras_model.get_layer('external_attention_rnn_wrapper_local_attention_v3_1')
        self._lstm_weight_i, self._lstm_weight_h, self._lstm_bias = self._attention_layer.get_weights()
        self._Wv = K.batch_get_value(self._attention_layer._Wv)
        self._Wg = K.batch_get_value(self._attention_layer._Wg)
        self._V = K.batch_get_value(self._attention_layer._V)
        self._Wx = K.batch_get_value(self._attention_layer._Wx)
        self._Wh = K.batch_get_value(self._attention_layer._Wh)
        self._Ws = K.batch_get_value(self._attention_layer._Ws)
        self._output_weight, self._output_bias = self._keras_model.get_layer('output').get_weights()

    def _relu(self, x):
        return np.maximum(x, 0)

    def _explain_CNN(self, X, relevance_value):
        relevance = self._CNN_explainer.analyze([X, relevance_value])
        return relevance

    def _lstm_forward(self, xt, htm1, ctm1, weight_i, weight_h, bias):
        z = np.dot(xt, weight_i)
        z += np.dot(htm1, weight_h)
        z = z + bias
        z0 = z[:, :self._hidden_dim]
        z1 = z[:, self._hidden_dim: 2 * self._hidden_dim]
        z2 = z[:, 2 * self._hidden_dim: 3 * self._hidden_dim]
        z3 = z[:, 3 * self._hidden_dim:]
        i = sigmoid(z0)
        f = sigmoid(z1)
        c = f * ctm1 + i * np.tanh(z2)
        o = sigmoid(z3)
        ht = o * np.tanh(c)
        ct = c
        return ht, ct, z2, i, f

    def _forward_beam_search(self, X, beam_search_captions):
        EOS_ENCODED = self._preprocessor.EOS_TOKEN_LABEL_ENCODED
        SOS_ENCODER = self._preprocessor.SOS_TOKEN_LABEL_ENCODED
        # print(EOS_ENCODED, SOS_ENCODER)
        self.caption = beam_search_captions
        img_input = X
        self._img_feature_input = self._image_model.predict(img_input)[0].reshape(self.L, self.D)
        # print('image feature input shape', self._img_feature_input.shape) # 196, 512
        self._image_features_before_act = np.zeros((self.L, self._hidden_dim))
        for i in range(self.L):
            self._image_features_before_act[i] = np.dot(self._img_feature_input[i],
                                                        self._image_features_wieght) + self._image_features_bias
        self._image_features = self._relu(self._image_features_before_act)
        self._average_img_feature = np.mean(self._img_feature_input, axis=0)  # 512,
        self._global_img_feature_before_act = np.dot(self._average_img_feature,
                                                     self._global_img_feature_weight) + self._global_img_feature_bias
        self._global_img_feature = self._relu(self._global_img_feature_before_act)  # 512,
        self._total_static_img_feature = np.dot(self._image_features, self._Wv)  # 196, 512
        self.ht = np.zeros((1, self._hidden_dim), dtype='float32')
        self.ct = np.zeros((1, self._hidden_dim), dtype='float32')
        self.gt = np.zeros((1, self._hidden_dim), dtype='float32')
        self.it_act = np.zeros((1, self._hidden_dim), dtype='float32')
        self.ft_act = np.zeros((1, self._hidden_dim), dtype='float32')
        self.context = np.zeros((1, self._hidden_dim), dtype='float32')
        self.st = np.zeros((1, self._hidden_dim), dtype='float32')
        self.attention = np.zeros((1, self.L), dtype='float32')
        self.beta = np.zeros((1, 1), dtype='float32')
        self.c_hat = np.zeros((1, self._hidden_dim), dtype='float32')
        for i in range(len(beam_search_captions)):
            htm1 = self.ht[-1]
            ctm1 = self.ct[-1]
            if i == 0:
                word_embedding = self._embedding.predict(np.array([SOS_ENCODER - 1]))[0]
                xt = np.hstack((word_embedding, self._global_img_feature.reshape(1, self._embedding_dim)))
                self.xt = np.hstack((word_embedding, self._global_img_feature.reshape(1, self._embedding_dim)))
            else:
                word = self._embedding.predict(np.array([beam_search_captions[i - 1] - 1]))[0]
                xt = np.hstack((word, self._global_img_feature.reshape(1, self._embedding_dim)))
                self.xt = np.vstack((self.xt, xt))
            ht, ct, gt, it_act, ft_act = self._lstm_forward(xt, htm1, ctm1, self._lstm_weight_i,
                                                            self._lstm_weight_h, self._lstm_bias)
            ht_proj = np.dot(ht, self._Wg)
            attention_before_sf = np.dot(np.tanh(ht_proj + self._total_static_img_feature, dtype='float32'), self._V)
            attention = softmax(attention_before_sf, axis=0)
            st = np.tanh(ct) * sigmoid(np.dot(xt, self._Wx) + np.dot(htm1, self._Wh))  # hidden_dim
            zt_extend = np.dot(np.tanh(np.dot(st, self._Ws) + ht_proj), self._V)
            alpha_beta_before_sf = np.concatenate((attention_before_sf, zt_extend), axis=0)
            beta = softmax(alpha_beta_before_sf, axis=0)[-1][0]
            context = np.reshape(np.sum(attention * self._image_features, axis=0), (1, self._hidden_dim))
            c_hat = beta * st + (1 - beta) * context
            ht_context = ht + c_hat
            caption_preds = np.dot(ht_context, self._output_weight) + self._output_bias
            self.ht = np.vstack((self.ht, ht.reshape(1, self._hidden_dim)))
            self.gt = np.vstack((self.gt, gt.reshape(1, self._hidden_dim)))
            self.it_act = np.vstack((self.it_act, it_act.reshape(1, self._hidden_dim)))
            self.ft_act = np.vstack((self.ft_act, ft_act.reshape(1, self._hidden_dim)))
            self.ct = np.vstack((self.ct, ct.reshape(1, self._hidden_dim)))
            self.context = np.vstack((self.context, context.reshape(1, self._hidden_dim)))
            self.attention = np.vstack((self.attention, attention.reshape(1, self.L)))
            self.st = np.vstack((self.st, st.reshape(1, self._hidden_dim)))
            self.beta = np.vstack((self.beta, beta.reshape(1, 1)))
            self.c_hat = np.vstack((self.c_hat, c_hat))
            if i == 0:
                self.caption_preds = caption_preds
            else:
                self.caption_preds = np.vstack((self.caption_preds, caption_preds))

    def _get_sign_stabilizer(self, z, eps):
        sign_z = np.ones(z.shape)
        sign_z[z < 0] = -1
        return z + sign_z * eps

    def _propagate_relevance_linear_lrp(self, r_in, forward_input, forward_output, bias, bias_nb_units,
                                        weight, bias_factor=0, eps=K.epsilon(), patterns=None):
        attribution_weight = weight
        forward_output_eps = self._get_sign_stabilizer(forward_output, eps)
        attribution = np.multiply(attribution_weight, forward_input[:, np.newaxis]) + (
                    (bias_factor * 1.0 * bias[np.newaxis, :]) / bias_nb_units)  # (D, O)
        attribution_norm = np.divide(attribution, forward_output_eps)  # (D, O)
        relevance = np.sum(attribution_norm * r_in, axis=1)  # (D, )
        return relevance

    def _explain_lstm_single_word_sequence(self, t=0, rule='eps'):
        if t > len(self.xt):
            raise NotImplementedError('index out of range of captions')
        explain_caption_encode = self.caption[t - 1] - 1  # real caption encode -1 is the encode used for prediciton
        explain_xht = np.hstack((self.xt[0:t], self.ht[0:t]))
        explain_relevance = np.zeros((1, len(self.caption_preds[0])))
        explain_relevance[0, explain_caption_encode] = self.caption_preds[t - 1, explain_caption_encode]
        r_V = np.zeros((self.L, self._hidden_dim), dtype='float32')
        r_img_feature_input = np.zeros((self.L, self.D), dtype='float32')
        weight_ig = np.split(self._lstm_weight_i, 4, 1)[2]  # (812, 512)
        weight_hg = np.split(self._lstm_weight_h, 4, 1)[2]  # (512, 512)
        weight_g = np.vstack((weight_ig, weight_hg))  # (600, 300)
        bias_g = np.split(self._lstm_bias, 4)[2]
        r_global_img_feature = np.zeros(self._embedding_dim)
        r_wording_embedding = np.zeros((t, self._embedding_dim))
        r_ct = np.zeros((t + 1, self._hidden_dim))
        r_ht = np.zeros((t + 1, self._hidden_dim))
        r_gt = np.zeros((t, self._hidden_dim))
        r_xht = np.zeros((t, self._embedding_dim * 2 + self._hidden_dim))

        self.relevance_rule = self._propagate_relevance_linear_lrp

        r_ht_context = self.relevance_rule(r_in=explain_relevance,
                                           forward_input=self.ht[t] + self.c_hat[t],
                                           forward_output=self.caption_preds[t - 1],
                                           bias=self._output_bias,
                                           bias_nb_units=self._hidden_dim,
                                           weight=self._output_weight)

        r_ht[t] = self.relevance_rule(r_in=r_ht_context,
                                      forward_input=self.ht[t],
                                      forward_output=self.ht[t] + self.c_hat[t],
                                      bias=np.zeros(self._hidden_dim),
                                      bias_nb_units=self._hidden_dim,
                                      weight=np.identity(self._hidden_dim))

        r_c_hat = self.relevance_rule(r_in=r_ht_context,
                                      forward_input=self.c_hat[t],
                                      forward_output=self.ht[t] + self.c_hat[t],
                                      bias=np.zeros(self._hidden_dim),
                                      bias_nb_units=self._hidden_dim,
                                      weight=np.identity(self._hidden_dim))

        r_context = self.relevance_rule(r_in=r_c_hat,
                                        forward_input=(1 - self.beta[t][0]) * self.context[t],
                                        forward_output=self.c_hat[t],
                                        bias=np.zeros(self._hidden_dim),
                                        bias_nb_units=self._hidden_dim,
                                        weight=np.identity(self._hidden_dim))
        r_st = self.relevance_rule(r_in=r_c_hat,
                                   forward_input=self.beta[t][0] * self.st[t],
                                   forward_output=self.c_hat[t],
                                   bias=np.zeros(self._hidden_dim),
                                   bias_nb_units=self._hidden_dim,
                                   weight=np.identity(self._hidden_dim))
        r_ct[t] = r_st

        for i in range(t)[::-1]:
            r_ct[i + 1] += r_ht[i + 1]
            # print('r_ct', r_ct[i+1])
            r_gt[i] = self.relevance_rule(r_in=r_ct[i + 1],
                                          forward_input=self.it_act[i + 1] * np.tanh(self.gt[i + 1]),
                                          forward_output=self.ct[i + 1],
                                          bias=np.zeros(self._hidden_dim),
                                          bias_nb_units=self._hidden_dim,
                                          weight=np.identity(self._hidden_dim))
            # print('r_gt', r_gt[i])
            r_ct[i] = self.relevance_rule(r_in=r_ct[i + 1],
                                          forward_input=self.ft_act[i + 1] * self.ct[i],
                                          forward_output=self.ct[i + 1],
                                          bias=np.zeros(self._hidden_dim),
                                          bias_nb_units=self._hidden_dim,
                                          weight=np.identity(self._hidden_dim))
            r_xht[i] = self.relevance_rule(r_in=r_gt[i],
                                           forward_input=explain_xht[i],
                                           forward_output=self.gt[i + 1],
                                           bias=bias_g,
                                           bias_nb_units=len(explain_xht[0]),
                                           weight=weight_g)
            # print('r_xht', r_xht[i])
            r_ht[i] = r_xht[i][self._embedding_dim * 2:]
            # print('r_ht', r_ht[i])
            r_global_img_feature += r_xht[i][self._embedding_dim:self._embedding_dim * 2]
            r_wording_embedding[i] = r_xht[i][:self._embedding_dim]
        r_average_img_feature = self.relevance_rule(r_in=r_global_img_feature,
                                                    forward_input=self._average_img_feature,
                                                    forward_output=self._global_img_feature_before_act,
                                                    bias=self._global_img_feature_bias,
                                                    bias_nb_units=self.D,
                                                    weight=self._global_img_feature_weight)

        for i in range(self.L):
            r_img_feature_input[i] = self.relevance_rule(r_in=r_average_img_feature,
                                                         forward_input=self._img_feature_input[i] / self.L,
                                                         forward_output=self._average_img_feature,
                                                         bias=np.zeros(self.D),
                                                         bias_nb_units=self.D,
                                                         weight=np.identity(self.D))
            r_V[i] = self.relevance_rule(r_in=r_context,
                                         forward_input=self._image_features[i] * self.attention[t][i],
                                         forward_output=self.context[t],
                                         bias=np.zeros(self._hidden_dim),
                                         bias_nb_units=self._hidden_dim,
                                         weight=np.identity(self._hidden_dim))
            r_img_feature_input[i] += self.relevance_rule(r_in=r_V[i],
                                                          forward_input=self._img_feature_input[i],
                                                          forward_output=self._image_features_before_act[i],
                                                          bias=self._image_features_bias,
                                                          bias_nb_units=self.D,
                                                          weight=self._image_features_wieght)
        self.r_words = np.sum(r_wording_embedding, axis=-1)
        self.r_words[0] = 0
        max_abso = np.max(np.abs(self.r_words))
        if max_abso:
            self.r_words = self.r_words / max_abso
        self.r_words = self.r_words[1:]
        return r_img_feature_input.reshape(1, int(np.sqrt(self.L)), int(np.sqrt(self.L)), self.D), self.attention[t]

    def call(self, inputs):
        def project(x):
            absmax = np.max(np.abs(x))
            if absmax == 0:
                return np.zeros(x.shape)
            x = 1.0 * x / absmax
            return x
        assert len(inputs) == 3
        caption_inputs = inputs[0]
        img_inputs = inputs[1]
        y_preds = inputs[2]
        vocab_length = y_preds.shape[-1]
        sentence_length = y_preds.shape[-2]
        lrp_inference_preds = np.zeros(y_preds.shape)
        caption_img_pairs = zip(caption_inputs, img_inputs, y_preds)
        idx = 0
        for item in caption_img_pairs:

            img_input = item[1][np.newaxis, :]
            caption_preds = item[2]
            caption_encoded = np.argmax(caption_preds, axis=-1)
            caption_encoded += 1
            self._forward_beam_search(img_input, caption_encoded)
            for i in range(sentence_length):
                lrp_inference = np.zeros(vocab_length)
                word_encode = caption_encoded[i]
                word = self._preprocessor._word_of[word_encode]
                if word in STOP_WORDS:
                    continue
                elif word_encode == self._EOS_ENCODED:
                    break
                else:
                    relevance, attention = self._explain_lstm_single_word_sequence(i + 1)
                    relevance = self._explain_CNN(img_input, relevance)
                    channels_first = K.image_data_format() == "channels_first"
                    hp = postprocess(relevance, self._color_conversion, channels_first)
                    hp = np.mean(hp, axis=-1)[0]
                    hp = project(hp)
                    if self._lrp_inference_mode == 'mean':
                        lrp_inference_score = np.mean(hp)
                    elif self._lrp_inference_mode == 'pos_mean':
                        lrp_inference_score = np.mean(np.maximum(hp, 0))
                    elif self._lrp_inference_mode == 'quantile':
                        lrp_inference_score = np.quantile(hp, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])[8]
                    else:
                        raise NotImplementedError("the lrp inference mode is not available")
                    lrp_inference[word_encode] = lrp_inference_score
                    lrp_inference_preds[idx][i] = lrp_inference
            idx += 1
        lrp_inference_preds = 1 + lrp_inference_preds
        return lrp_inference_preds
class LRPInferenceLayergridTD(object):

    def __init__(self, model, dataset_provider, hidden_dim, embedding_dim, L, D, img_encoder, lrp_inference_mode):
        self._preprocessor = dataset_provider.caption_preprocessor
        self._EOS_ENCODED = self._preprocessor.EOS_TOKEN_LABEL_ENCODED
        self._SOS_ENCODER = self._preprocessor.SOS_TOKEN_LABEL_ENCODED
        self._dataset_provider = dataset_provider
        self._max_caption_length = 20
        self._hidden_dim = hidden_dim
        self._embedding_dim = embedding_dim
        self._lrp_inference_mode = lrp_inference_mode
        self.L = L
        self.D = D
        self._img_encoder = img_encoder
        if self._img_encoder in ['vgg16', 'vgg19']:
            self._color_conversion = 'BGRtoRGB'
            self._reshape_size = (14, 14)
            self._upscale = 16
        self._keras_model = model
        self._image_model = Model(inputs=self._keras_model.get_layer('input_1').input,
                                  outputs=self._keras_model.get_layer('block5_conv3').output)
        self._CNN_explainer = LRPSequentialPresetA(self._image_model, epsilon=EPS, neuron_selection_mode='replace')

        self._img_feature_input_model_bm = Model(inputs=self._keras_model.get_layer('input_1').input,
                                        outputs=self._keras_model.get_layer('reshape_1').output)
        self._image_features_weight_bm, self._image_features_bias_bm = self._keras_model.get_layer(
            'image_features').get_weights()
        self._embedding_bm = Model(self._keras_model.get_layer('embedding_1').input,
                                self._keras_model.get_layer('embedding_1').output)
        self._attention_layer_bm = self._keras_model.get_layer('external_bottom_up_attention_adaptive_1')
        self._image_features_wieght, self._image_features_bias = self._keras_model.get_layer(
            'image_features').get_weights()
        self._global_img_feature_weight_bm, self._global_img_feature_bias_bm = self._keras_model.get_layer(
            'global_img_feature').get_weights()
        self._language_lstm_weight_i, self._language_lstm_weight_h, self._language_lstm_bias = self._attention_layer_bm.get_weights()
        self._W_va = K.batch_get_value(self._attention_layer_bm._W_va)
        self._W_ha = K.batch_get_value(self._attention_layer_bm._W_ha)
        self._W_a = K.batch_get_value(self._attention_layer_bm._W_a)
        self._top_down_lstm_weight_i = K.batch_get_value(self._attention_layer_bm._top_down_lstm_weight_i)
        self._top_down_lstm_weight_h = K.batch_get_value(self._attention_layer_bm._top_down_lstm_weight_h)
        self._top_down_lstm_weight_bias = K.batch_get_value(self._attention_layer_bm._top_down_lstm_weight_bias)

        self._output_weight_bm, self._output_bias_bm = self._keras_model.get_layer('output').get_weights()
        self._W_x = K.batch_get_value(self._attention_layer_bm._W_x)
        self._W_s = K.batch_get_value(self._attention_layer_bm._W_s)
        self._W_h = K.batch_get_value(self._attention_layer_bm._W_h)

    def _relu(self, x):
        return np.maximum(x, 0)

    def _explain_CNN(self, X, relevance_value):
        relevance = self._CNN_explainer.analyze([X, relevance_value])
        return relevance

    def _lstm_forward(self, xt, htm1, ctm1, weight_i, weight_h, bias):
        z = np.dot(xt, weight_i)
        z += np.dot(htm1, weight_h)
        z = z + bias
        z0 = z[:, :self._hidden_dim]
        z1 = z[:, self._hidden_dim: 2 * self._hidden_dim]
        z2 = z[:, 2 * self._hidden_dim: 3 * self._hidden_dim]
        z3 = z[:, 3 * self._hidden_dim:]
        i = sigmoid(z0)
        f = sigmoid(z1)
        c = f * ctm1 + i * np.tanh(z2)
        o = sigmoid(z3)
        ht = o * np.tanh(c)
        ct = c
        return ht, ct, z2, i, f

    def _forward_beam_search(self, X, beam_search_captions):
        EOS_ENCODED = self._preprocessor.EOS_TOKEN_LABEL_ENCODED
        SOS_ENCODER = self._preprocessor.SOS_TOKEN_LABEL_ENCODED
        img_input = X
        self.caption = beam_search_captions
        self._image_features_input_bm = self._img_feature_input_model_bm.predict(img_input)[0]
        self._average_img_feature_bm = np.mean(self._image_features_input_bm, axis=0)
        self._global_image_feature_before_act_bm = np.dot(self._average_img_feature_bm, self._global_img_feature_weight_bm) + self._global_img_feature_bias_bm
        self._global_image_feature_bm = self._relu(self._global_image_feature_before_act_bm)

        self._image_features_before_act_bm = np.zeros((self.L, self._hidden_dim))
        for i in range(self.L):
            self._image_features_before_act_bm[i] = np.dot(self._image_features_input_bm[i],
                                                        self._image_features_weight_bm) + self._image_features_bias_bm
        self._image_features_bm = self._relu(self._image_features_before_act_bm)
        self._image_features_proj_bm = np.dot(self._image_features_bm, self._W_va)

        self.h1t = np.zeros((1, self._hidden_dim), dtype='float32')
        self.c1t = np.zeros((1, self._hidden_dim), dtype='float32')
        self.h2t = np.zeros((1, self._hidden_dim), dtype='float32')
        self.c2t = np.zeros((1, self._hidden_dim), dtype='float32')
        self.g1t = np.zeros((1, self._hidden_dim), dtype='float32')
        self.i1t_act = np.zeros((1, self._hidden_dim), dtype='float32')
        self.f1t_act = np.zeros((1, self._hidden_dim), dtype='float32')
        self.g2t = np.zeros((1, self._hidden_dim), dtype='float32')
        self.i2t_act = np.zeros((1, self._hidden_dim), dtype='float32')
        self.f2t_act = np.zeros((1, self._hidden_dim), dtype='float32')
        self.x2t = np.zeros((1, self._hidden_dim * 2))
        self.x1t = np.zeros((1, self._hidden_dim + self._embedding_dim + self._embedding_dim))
        self.context = np.zeros((1, self._hidden_dim))
        self.st = np.zeros((1, self._hidden_dim))
        self.beta = np.zeros((1,1))
        self.context_hat = np.zeros((1, self._hidden_dim))
        self.attention = np.zeros((1, self.L))
        for i in range(len(beam_search_captions)):
            h1tm1 = self.h1t[-1].reshape(1, self._hidden_dim)
            c1tm1 = self.c1t[-1].reshape(1, self._hidden_dim)
            h2tm1 = self.h2t[-1].reshape(1, self._hidden_dim)
            c2tm1 = self.c2t[-1].reshape(1, self._hidden_dim)
            if i == 0:
                word_embedding = self._embedding_bm.predict(np.array([SOS_ENCODER - 1]))[0]
                # print('word_embedding', word_embedding.dtype)
                x1t = np.hstack((h2tm1, self._global_image_feature_bm.reshape(1, self._embedding_dim), word_embedding))
                # print('xt', xt.dtype)
            else:
                word_embedding = self._embedding_bm.predict(np.array([beam_search_captions[i-1]-1]))[0]
                x1t = np.hstack((h2tm1, self._global_image_feature_bm.reshape(1, self._embedding_dim), word_embedding))
                # print('xt', xt.dtype)
            h1t, c1t, g1t, i1t_act, f1t_act = self._lstm_forward(x1t, h1tm1, c1tm1, self._top_down_lstm_weight_i,
                                                                 self._top_down_lstm_weight_h,
                                                                 self._top_down_lstm_weight_bias)
            h_proj = np.dot(h1t, self._W_ha)
            attention_bf_act = np.dot(np.tanh(self._image_features_proj_bm + h_proj), self._W_a)
            attention = softmax(attention_bf_act, axis=0)
            context = np.reshape(np.sum(attention * self._image_features_bm, axis=0), (1, self._hidden_dim))

            st = np.tanh(c1t) * sigmoid(np.dot(x1t, self._W_x) + np.dot(h1tm1, self._W_h))
            zt_extend = np.dot(np.tanh(np.dot(st, self._W_s) + h_proj), self._W_a)
            alpha_beta_before_sf = np.concatenate((attention_bf_act, zt_extend), axis=0)
            beta = softmax(alpha_beta_before_sf, axis=0)[-1][0]
            context_hat = beta * st + (1 - beta) * context

            x2t = np.hstack((context_hat, h1t.reshape(1, self._hidden_dim)))
            h2t, c2t, g2t, i2t_act, f2t_act = self._lstm_forward(x2t, h2tm1, c2tm1, self._language_lstm_weight_i,
                                                                 self._language_lstm_weight_h, self._language_lstm_bias)
            caption_preds = np.dot(h2t, self._output_weight_bm) + self._output_bias_bm
            self.h1t = np.vstack((self.h1t, h1t.reshape(1, self._hidden_dim)))
            self.h2t = np.vstack((self.h2t, h2t.reshape(1, self._hidden_dim)))
            self.c1t = np.vstack((self.c1t, c1t.reshape(1, self._hidden_dim)))
            self.c2t = np.vstack((self.c2t, c2t.reshape(1, self._hidden_dim)))
            self.x1t = np.vstack((self.x1t, x1t))
            self.x2t = np.vstack((self.x2t, x2t))
            self.g1t = np.vstack((self.g1t, g1t.reshape(1, self._hidden_dim)))
            self.i1t_act = np.vstack((self.i1t_act, i1t_act.reshape(1, self._hidden_dim)))
            self.f1t_act = np.vstack((self.f1t_act, f1t_act.reshape(1, self._hidden_dim)))
            self.g2t = np.vstack((self.g2t, g2t.reshape(1, self._hidden_dim)))
            self.i2t_act = np.vstack((self.i2t_act, i2t_act.reshape(1, self._hidden_dim)))
            self.f2t_act = np.vstack((self.f2t_act, f2t_act.reshape(1, self._hidden_dim)))
            self.context = np.vstack((self.context, context.reshape(1, self._hidden_dim)))
            self.st = np.vstack((self.st, st.reshape(1, self._hidden_dim)))
            self.beta = np.vstack((self.beta, beta.reshape(1,1)))
            self.context_hat = np.vstack((self.context_hat, context_hat.reshape(1, self._hidden_dim)))
            self.attention = np.vstack((self.attention, attention.reshape(1, self.L)))

            if i == 0:
                self.caption_preds = caption_preds
            else:
                self.caption_preds = np.vstack((self.caption_preds, caption_preds))
        self.x1t = self.x1t[1:]
        self.x2t = self.x2t[1:]

    def _explain_lstm_single_word_sequence(self, t=0):
        if t > len(self.x1t):
            raise NotImplementedError('index out of range of captions')
        explain_caption_encode = self.caption[t - 1] - 1
        explain_relevance = np.zeros((1, len(self.caption_preds[0])))
        explain_relevance[0, explain_caption_encode] = self.caption_preds[t - 1, explain_caption_encode]

        explain_xht1 = np.hstack((self.x1t[0:t], self.h1t[0:t]))
        explain_xht2 = np.hstack((self.x2t[0:t], self.h2t[0:t]))
        r_V = np.zeros((self.L, self._hidden_dim), dtype='float32')
        r_img_feature_input = np.zeros((self.L, self.D), dtype='float32')
        r_global_img_feature = np.zeros(self._embedding_dim)
        r_c1t = np.zeros((t+1, self._hidden_dim))
        r_c2t = np.zeros((t+1, self._hidden_dim))
        r_h1t = np.zeros((t+1, self._hidden_dim))
        r_h2t = np.zeros((t+1, self._hidden_dim))
        r_g1t = np.zeros((t, self._hidden_dim))
        r_g2t = np.zeros((t, self._hidden_dim))
        r_xht1 = np.zeros((t, self._embedding_dim + self._embedding_dim + self._hidden_dim + self._hidden_dim)) # h2t-globalimgfeature-wordembedding-ht1
        r_xht2 = np.zeros((t, self._hidden_dim * 3))
        r_context = np.zeros((t, self._hidden_dim))
        r_context_hat = np.zeros((t, self._hidden_dim))
        r_wordembedding = np.zeros((t, self._embedding_dim))
        top_down_weight_ig = np.split(self._top_down_lstm_weight_i, 4, 1)[2]  # (812, 512)
        top_down_weight_hg = np.split(self._top_down_lstm_weight_h, 4, 1)[2]  # (512, 512)
        top_down_weight_g = np.vstack((top_down_weight_ig, top_down_weight_hg))  # (600, 300)
        top_down_bias_g = np.split(self._top_down_lstm_weight_bias, 4)[2]
        language_weight_ig = np.split(self._language_lstm_weight_i, 4, 1)[2]
        language_weight_hg = np.split(self._language_lstm_weight_h, 4, 1)[2]
        language_weight_g = np.vstack((language_weight_ig, language_weight_hg))
        language_bias_g = np.split(self._language_lstm_bias, 4)[2]

        r_h2t_context_hat_predict = self._propagate_relevance_linear_lrp(r_in=explain_relevance,
                                                        forward_output=self.caption_preds[t-1],
                                                        forward_input=self.h2t[t] + self.context_hat[t],
                                                        bias=self._output_bias_bm,
                                                        bias_nb_units=self._hidden_dim,
                                                        weight=self._output_weight_bm)
        r_h2t[t] = self._propagate_relevance_linear_lrp(r_in=r_h2t_context_hat_predict,
                                                        forward_output=self.h2t[t] + self.context_hat[t],
                                                        forward_input=self.h2t[t],
                                                        bias=np.zeros(self._hidden_dim),
                                                        bias_nb_units=self._hidden_dim,
                                                        weight=np.identity(self._hidden_dim))
        r_context_hat[t-1] = self._propagate_relevance_linear_lrp(r_in=r_h2t_context_hat_predict,
                                                             forward_output=self.h2t[t] + self.context_hat[t],
                                                             forward_input=self.context_hat[t],
                                                             bias=np.zeros(self._hidden_dim),
                                                             bias_nb_units=self._hidden_dim,
                                                             weight=np.identity(self._hidden_dim))


        for i in range(t)[::-1]:
            r_c2t[i+1] += r_h2t[i+1]
            r_g2t[i] = self._propagate_relevance_linear_lrp(r_in=r_c2t[i+1],
                                                            forward_output=self.c2t[i+1],
                                                            forward_input=self.i2t_act[i+1] * np.tanh(self.g2t[i+1]),
                                                            bias=np.zeros(self._hidden_dim),
                                                            bias_nb_units=self._hidden_dim,
                                                            weight=np.identity(self._hidden_dim))
            r_c2t[i] = self._propagate_relevance_linear_lrp(r_in=r_c2t[i+1],
                                                            forward_output=self.c2t[i+1],
                                                            forward_input=self.f2t_act[i+1] * self.c2t[i],
                                                            bias=np.zeros(self._hidden_dim),
                                                            bias_nb_units=self._hidden_dim,
                                                            weight=np.identity(self._hidden_dim))
            r_xht2[i] = self._propagate_relevance_linear_lrp(r_in=r_g2t[i],
                                                             forward_output=self.g2t[i+1],
                                                             forward_input=explain_xht2[i],
                                                             bias=language_bias_g,
                                                             bias_nb_units=self._hidden_dim * 3,
                                                             weight=language_weight_g)
            r_h1t[i+1] += r_xht2[i][self._hidden_dim: self._hidden_dim*2]
            r_h2t[i] += r_xht2[i][self._hidden_dim*2:]
            r_context_hat[i] += r_xht2[i][:self._hidden_dim]
            r_st = self._propagate_relevance_linear_lrp(r_in=r_context_hat[i],
                                                        forward_output=self.context_hat[i+1],
                                                        forward_input=self.beta[i+1][0] * self.st[i+1],
                                                        bias=np.zeros(self._hidden_dim),
                                                        bias_nb_units=self._hidden_dim,
                                                        weight=np.identity(self._hidden_dim))
            r_context[i] = self._propagate_relevance_linear_lrp(r_in=r_context_hat[i],
                                                                forward_output=self.context_hat[i+1],
                                                                forward_input=self.context[i+1] * (1 - self.beta[i+1][0]),
                                                                bias=np.zeros(self._hidden_dim),
                                                                bias_nb_units=self._hidden_dim,
                                                                weight=np.identity(self._hidden_dim))

            r_c1t[i+1] += r_st
            r_c1t[i+1] += r_h1t[i+1]
            r_g1t[i] = self._propagate_relevance_linear_lrp(r_in=r_c1t[i+1],
                                                            forward_output=self.c1t[i+1],
                                                            forward_input=self.i1t_act[i+1] * np.tanh(self.g1t[i+1]),
                                                            bias=np.zeros(self._hidden_dim),
                                                            bias_nb_units=self._hidden_dim,
                                                            weight=np.identity(self._hidden_dim))
            r_c1t[i] = self._propagate_relevance_linear_lrp(r_in=r_c1t[i+1],
                                                            forward_output=self.c1t[i+1],
                                                            forward_input=self.f1t_act[i+1] * self.c1t[i],
                                                            bias=np.zeros(self._hidden_dim),
                                                            bias_nb_units=self._hidden_dim,
                                                            weight=np.identity(self._hidden_dim))
            r_xht1[i] = self._propagate_relevance_linear_lrp(r_in=r_g1t[i],
                                                             forward_output=self.g1t[i+1],
                                                             forward_input=explain_xht1[i],
                                                             bias=top_down_bias_g,
                                                             bias_nb_units=self._hidden_dim * 2 + self._embedding_dim*2,
                                                             weight=top_down_weight_g)
            r_h2t[i] += r_xht1[i][:self._hidden_dim]

            r_global_img_feature += r_xht1[i][self._hidden_dim:self._embedding_dim+self._hidden_dim]
            r_wordembedding[i] = r_xht1[i][self._hidden_dim + self._embedding_dim:self._embedding_dim * 2 + self._hidden_dim]
            # if i >= t-5:
            for k in range(self.L):
                r_V[k] += self._propagate_relevance_linear_lrp(r_in=r_context[i],
                                                               forward_input=self._image_features_bm[k] *
                                                                             self.attention[i+1][k],
                                                               forward_output=self.context[i+1],
                                                               bias=np.zeros(self._hidden_dim),
                                                               bias_nb_units=self._hidden_dim,
                                                               weight=np.identity(self._hidden_dim))
            r_h1t[i] += r_xht1[i][self._embedding_dim * 2 + self._hidden_dim:]
        r_average_img_feature = self._propagate_relevance_linear_lrp(r_in=r_global_img_feature,
                                                                     forward_input=self._average_img_feature_bm,
                                                                     forward_output=self._global_image_feature_before_act_bm,
                                                                     bias=self._global_img_feature_bias_bm,
                                                                     bias_nb_units=self.D,
                                                                     weight=self._global_img_feature_weight_bm)

        for i in range(self.L):
            r_img_feature_input[i] = self._propagate_relevance_linear_lrp(r_in=r_average_img_feature,
                                                                          forward_input=self._image_features_input_bm[i] / self.L,
                                                                          forward_output=self._average_img_feature_bm,
                                                                          bias=np.zeros(self.D),
                                                                          bias_nb_units=self.D,
                                                                          weight=np.identity(self.D))
            r_img_feature_input[i] += self._propagate_relevance_linear_lrp(r_in=r_V[i],
                                                                           forward_input=self._image_features_input_bm[i],
                                                                           forward_output=self._image_features_before_act_bm[i],
                                                                           bias=self._image_features_bias_bm,
                                                                           bias_nb_units=self.D,
                                                                           weight=self._image_features_weight_bm)
        self.r_words = np.sum(r_wordembedding, axis=-1)
        return r_img_feature_input.reshape(1, int(np.sqrt(self.L)), int(np.sqrt(self.L)), self.D), self.attention[t]

    def _get_sign_stabilizer(self, z, eps):
        sign_z = np.ones(z.shape)
        sign_z[z < 0] = -1
        return z + sign_z * eps

    def _propagate_relevance_linear_lrp(self, r_in, forward_input, forward_output, bias, bias_nb_units,
                                        weight, bias_factor=0, eps=K.epsilon()):
        attribution_weight = weight
        forward_output_eps = self._get_sign_stabilizer(forward_output, eps)
        attribution = np.multiply(attribution_weight, forward_input[:, np.newaxis]) + (
                    (bias_factor * 1.0 * bias[np.newaxis, :]) / bias_nb_units)  # (D, O)
        attribution_norm = np.divide(attribution, forward_output_eps)  # (D, O)
        relevance = np.sum(attribution_norm * r_in, axis=1)  # (D, )
        return relevance

    def call(self, inputs):
        def project(x):
            absmax = np.max(np.abs(x))
            if absmax == 0:
                return np.zeros(x.shape)
            x = 1.0 * x / absmax
            return x
        assert len(inputs) == 3
        caption_inputs = inputs[0]
        img_inputs = inputs[1]
        y_preds = inputs[2]
        vocab_length = y_preds.shape[-1]
        sentence_length = y_preds.shape[-2]
        lrp_inference_preds = np.zeros(y_preds.shape)
        caption_img_pairs = zip(caption_inputs, img_inputs, y_preds)
        idx = 0
        for item in caption_img_pairs:
            img_input = item[1][np.newaxis, :]
            caption_preds = item[2]
            caption_encoded = np.argmax(caption_preds, axis=-1)
            caption_encoded += 1
            self._forward_beam_search(img_input, caption_encoded)
            for i in range(sentence_length):
                lrp_inference = np.zeros(vocab_length)
                word_encode = caption_encoded[i]
                word = self._preprocessor._word_of[word_encode]
                if word in STOP_WORDS:
                    continue
                elif word_encode == self._EOS_ENCODED:
                    break
                else:
                    relevance, attention = self._explain_lstm_single_word_sequence(i + 1)
                    relevance = self._explain_CNN(img_input, relevance)
                    channels_first = K.image_data_format() == "channels_first"
                    hp = postprocess(relevance, self._color_conversion, channels_first)
                    hp = np.mean(hp, axis=-1)[0]
                    hp = project(hp)
                    if self._lrp_inference_mode == 'mean':
                        lrp_inference_score = np.mean(hp)
                    elif self._lrp_inference_mode == 'pos_mean':
                        lrp_inference_score = np.mean(np.maximum(hp, 0))
                    elif self._lrp_inference_mode == 'quantile':
                        lrp_inference_score = np.quantile(hp, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])[8]
                    else:
                        raise NotImplementedError("the lrp inference mode is not available")
                    lrp_inference[word_encode] = lrp_inference_score
                    lrp_inference_preds[idx][i] = lrp_inference
            idx += 1
        lrp_inference_preds = 1 + lrp_inference_preds
        return lrp_inference_preds
