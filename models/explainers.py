from models.preparedataset import Flickr30kDataset, DatasetPreprocessorAttention, COCODataset
from models.model import *
import config
from keras import backend as K
import os
from scipy.special import expit as sigmoid
from scipy.special import softmax
import skimage.transform
from inference import BatchNLargest, Caption
from keras.models import Model
from innvestigate.analyzer.relevance_based.relevance_analyzer import *
from innvestigate.analyzer.gradient_based import *
from innvestigate.examples.utils_imagenet import *
import matplotlib.pyplot as plt
from operator import attrgetter
from PIL import Image

EPS = 0.01
ALPHA =1
BETA = 0

class ExplainImgCaptioningAttentionModel(object):

    def __init__(self,  model, weight_path, dataset_provider, max_caption_length):

        self._keras_model = model.keras_model
        self._keras_model.load_weights(weight_path)

        self._image_model = Model(inputs=self._keras_model.get_layer('input_1').input,
                                  outputs=self._keras_model.get_layer('block5_conv3').output)
        self._img_encoder = model.img_encoder
        self._CNN_explainer = LRPSequentialPresetA(self._image_model, epsilon=EPS,  neuron_selection_mode='replace')
        self._preprocessor = dataset_provider.caption_preprocessor
        self._dataset_provider = dataset_provider
        self._max_caption_length = max_caption_length
        self._hidden_dim = model._hidden_dim
        self._embedding_dim = model._embedding_dim
        self.L = model.L
        self.D = model.D
        self._weight_path = weight_path.strip('hdf5')

    def _sigmoid(self, x):
        s = 1 / (1 + np.exp(-x))
        return s

    def _log_softmax(self, x):
        x = x - np.max(x, axis=-1, keepdims=True)  # For numerical stability

        return x - np.log(np.sum(np.exp(x), axis=-1, keepdims=True))

    def _beam_search(self, X, beam_size):
        _, imgs_input = X
        batch_size = imgs_input.shape[0]

        EOS_ENCODED = self._preprocessor.EOS_TOKEN_LABEL_ENCODED
        SOS_ENCODED = self._preprocessor.SOS_TOKEN_LABEL_ENCODED
        complete_captions = BatchNLargest(batch_size=batch_size,
                                          n=beam_size)
        partial_captions = BatchNLargest(batch_size=batch_size,
                                         n=beam_size)
        partial_captions.add([Caption(sentence_encoded=[SOS_ENCODED, EOS_ENCODED],
                                      log_prob=0.0)
                              for __ in range(batch_size)])

        for _ in range(self._max_caption_length):
            partial_captions_prev = partial_captions
            partial_captions = BatchNLargest(batch_size=batch_size,
                                             n=beam_size)

            for top_captions in partial_captions_prev.n_largest():
                sentences_encoded = [x.sentence_encoded for x in top_captions]
                captions_input, _ = self._preprocessor.preprocess_batch(sentences_encoded)
                preds = self._keras_model.predict_on_batch([captions_input, imgs_input])
                preds = self._log_softmax(preds)
                preds = preds[:, :-1, :]  # Discard the last word (dummy) comment this line if use attention model. attention model doesn't add the dummy word
                preds = preds[:, -1]  # We only care the last word in a caption
                top_words = np.argpartition(
                                preds, -beam_size)[:, -beam_size:]
                row_indexes = np.arange(batch_size)[:, np.newaxis]
                top_words_log_prob = preds[row_indexes, top_words]
                log_probs_prev = np.array([x.log_prob for x in top_captions])[:, np.newaxis]
                log_probs_total = top_words_log_prob + log_probs_prev

                partial_captions_result = []
                complete_captions_result = []
                for sentence, words, log_probs in zip(sentences_encoded,
                                                      top_words,
                                                      log_probs_total):
                    partial_captions_batch = []
                    complete_captions_batch = []
                    for word, log_prob in zip(words, log_probs):
                        word += 1  # Convert from model's to Tokenizer's
                        sentence_encoded = sentence[:-1] + [word, sentence[-1]]
                        caption = Caption(sentence_encoded=sentence_encoded,
                                          log_prob=log_prob)
                        partial_captions_batch.append(caption)
                        if word == EOS_ENCODED:
                            complete_caption = Caption(
                                                    sentence_encoded=sentence,
                                                    log_prob=log_prob)
                            complete_captions_batch.append(complete_caption)
                        else:
                            complete_captions_batch.append(None)
                    partial_captions_result.append(partial_captions_batch)
                    complete_captions_result.append(complete_captions_batch)
                partial_captions.add_many(partial_captions_result)
                complete_captions.add_many(complete_captions_result)
        top_partial_captions = partial_captions.n_largest(sort=True)[0]
        try:
            top_complete_captions = complete_captions.n_largest(sort=True)[0]
        except:
            top_complete_captions = [None] * beam_size
        results = []
        for partial_caption, complete_caption in zip(top_partial_captions,
                                                     top_complete_captions):
            if complete_caption is None:
                results.append(partial_caption.sentence_encoded[1:])
            else:
                results.append(complete_caption.sentence_encoded[1:])
        return results

    def _relu(self,x):
        return np.maximum(x, 0)

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

    def _get_sign_stabilizer(self, z, eps):
        sign_z = np.ones(z.shape)
        sign_z[z < 0] = -1
        return z + sign_z * eps

    def _forward_single(self, X):
        '''This function is a forward pass using numpy, which saves all the intermediate variables
        we also use this function to check if the numpy implementation generates the same captions as the keras forward with beam size 1'''
        raise NotImplementedError()

    def _forward_beam_search(self, X, beam_search_captions):
        '''This function save the intermediate variables given a predicted beam_search_captions
        The saved variables are further used for LRP explanations'''
        raise NotImplementedError()

    def _propagate_relevance_linear_lrp(self, r_in, forward_input, forward_output, bias, bias_nb_units,
                                           weight,  bias_factor=0, eps=K.epsilon(), patterns=None):

        '''LRP function for linear layer'''
        attribution_weight = weight
        forward_output_eps = self._get_sign_stabilizer(forward_output, eps)
        attribution = np.multiply(attribution_weight, forward_input[:, np.newaxis]) + ((bias_factor * 1.0*bias[np.newaxis,:]) / bias_nb_units)# (D, O)
        attribution_norm = np.divide(attribution, forward_output_eps)  # (D, O)
        relevance = np.sum(attribution_norm * r_in, axis=1)  # (D, )
        return relevance

    def _explain_lstm_single_word(self, t=0):
        '''t: the index of the word to explain start from 1
        return: relevance of image_encode'''
        # print(rule)

        raise NotImplementedError()

    def _explain_lstm_single_word_sequence(self, t=0):
        '''t: the index of the word to explain start from 1
        return: relevance of image_encode'''
        raise NotImplementedError()

    def _explain_CNN(self, X, relevance_value):
        relevance = self._CNN_explainer.analyze([X, relevance_value])
        return relevance

    def _explain_sentence(self):
        # print(rule)
        image_feature_relevance = []
        for i in range(len(self.caption)-1):
            relevance, attention = self._explain_lstm_single_word_sequence(i+1)
            image_feature_relevance.append(relevance)
        return image_feature_relevance, self.attention[1:-1]

    def process_beam_search(self, beam_size=3, ):
        data_generator = self._dataset_provider.test_set(include_datum=True)
        count = 0
        for X, y, data in data_generator:
            imgs_path = map(attrgetter('img_path'), data)
            imgs_path = list(imgs_path)
            count += 1
            if count == 1:  # 9
                beam_search_captions_encoded = self._beam_search(X, beam_size)[0]
                beam_search_captions = self._preprocessor.decode_captions_from_list1d(beam_search_captions_encoded)
                print(beam_search_captions)
                img_filename = imgs_path[0].split('/')[-1]
                save_folder = os.path.join(self._weight_path + 'explanation', img_filename.split('.')[0])
                if not os.path.isdir(save_folder):
                    os.makedirs(save_folder)
                img_original = Image.open(imgs_path[0])
                img_original = img_original.resize((224, 224))
                img_original.show()
                img_original.save(os.path.join(save_folder, img_filename))
                sequence_input, img_input, = X
                self._forward_beam_search(X, beam_search_captions_encoded)
                '''for a full sentence'''
                img_encode_relevance, attention = self._explain_sentence()
                x = int(np.sqrt(len(attention)))
                y = int(np.ceil(len(attention) / x))
                _, axes = plt.subplots(y, x, sharex="col", sharey="row", figsize=(20,20))
                # plt.axis('off')
                axes = axes.flatten()
                for i in range(len(attention)):
                    relevance = self._explain_CNN(img_input, img_encode_relevance[i])
                    # print(beam_search_captions[0].split()[i])
                    # print(np.sum(relevance[0] == 0))
                    # print(np.sum(relevance[0] > 0))
                    # print(np.sum(relevance[0] < 0))
                    channels_first = K.image_data_format() == "channels_first"
                    color_conversion = "BGRtoRGB" if "BGR" == "BGR" else None
                    hp = postprocess(relevance, color_conversion, channels_first)
                    hp = heatmap(hp)[0]
                    axes[i].set_title(self._preprocessor._word_of[beam_search_captions_encoded[i]], fontsize=18)
                    axes[i].imshow(hp)
                plt.savefig(os.path.join(save_folder, img_filename.split('.')[0] + 'lrp_hm.jpg'))
                plt.show()
                x = int(np.sqrt(len(attention)))
                y = int(np.ceil(len(attention) / x))
                _, axes = plt.subplots(y, x, sharex="col", sharey="row", figsize=(20,20))
                # plt.axis('off')
                axes = axes.flatten()
                for i in range(len(attention)):
                    if self._img_encoder == "inception_v3":
                        atn = skimage.transform.pyramid_expand(attention[i].reshape(8,8), upscale=37, sigma=20, multichannel=False)
                        atn = skimage.transform.resize(atn, (299,299), mode='reflect', anti_aliasing=True)
                        blank = np.zeros((299, 299, 2))
                    else:
                        atn = skimage.transform.pyramid_expand(attention[i].reshape(14, 14), upscale=16, sigma=20,
                                                               multichannel=False)
                        blank = np.zeros((224, 224, 2))
                    atn = atn[:, :, np.newaxis]
                    atn = (atn - np.min(atn)) / np.max(atn) * 255
                    atn = np.concatenate((atn, blank), axis=-1)
                    attention_img = Image.fromarray(np.uint8(atn), img_original.mode)
                    tmp_img = Image.blend(img_original, attention_img, 0.7)
                    axes[i].set_title(self._preprocessor._word_of[beam_search_captions_encoded[i]], fontsize=18)
                    axes[i].imshow(tmp_img)
                plt.savefig(os.path.join(save_folder, img_filename.split('.')[0] + 'attention.jpg'))
                plt.show()
                break



class ExplainImgCaptioningAdaptiveAttention(ExplainImgCaptioningAttentionModel):

    def __init__(self,  model, weight_path, dataset_provider, max_caption_length=20):
        super(ExplainImgCaptioningAdaptiveAttention, self).__init__(model, weight_path, dataset_provider, max_caption_length)
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
    def _forward_single(self, X):

        EOS_ENCODED = self._preprocessor.EOS_TOKEN_LABEL_ENCODED
        SOS_ENCODER = self._preprocessor.SOS_TOKEN_LABEL_ENCODED
        sequence_input, img_input = X
        self._img_feature_input = self._image_model.predict(img_input)[0].reshape(self.L, self.D)
        # print('image feature input shape', self._img_feature_input.shape) # 196, 512
        self._image_features_before_act = np.zeros((self.L, self._hidden_dim))
        for i in range(self.L):
            self._image_features_before_act[i] = np.dot(self._img_feature_input[i],
                                                        self._image_features_wieght) + self._image_features_bias
        self._image_features = self._relu(self._image_features_before_act)
        self._average_img_feature = np.mean(self._img_feature_input, axis=0)  # 512,
        # print('average_img_feautre', self._average_img_feature.shape)
        self._global_img_feature_before_act = np.dot(self._average_img_feature,
                                                     self._global_img_feature_weight) + self._global_img_feature_bias
        self._global_img_feature = self._relu(self._global_img_feature_before_act)  # 512,
        # print('global feature shape', self._global_img_feature.shape)
        self._total_static_img_feature = np.dot(self._image_features, self._Wv)  # 196, 512

        self.ht = np.zeros((1, self._hidden_dim), dtype='float32')
        self.ct = np.zeros((1, self._hidden_dim), dtype='float32')
        self.gt = np.zeros((1, self._hidden_dim), dtype='float32')
        self.it_act = np.zeros((1, self._hidden_dim), dtype='float32')
        self.ft_act = np.zeros((1, self._hidden_dim), dtype='float32')
        self.caption = []
        for i in range(self._max_caption_length):
            htm1 = self.ht[-1].reshape(1, self._hidden_dim)
            ctm1 = self.ct[-1].reshape(1, self._hidden_dim)
            if i == 0:
                word_embedding = self._embedding.predict(np.array([SOS_ENCODER]))[0]
                # print('word_embedding', word_embedding.dtype)
                xt = np.hstack((word_embedding, self._global_image_feature.reshape(1, self._hidden_dim)))
                # print('xt', xt.dtype)
                self.xt = np.hstack((word_embedding, self._global_image_feature.reshape(1, self._hidden_dim)))
            else:
                word = self._embedding.predict(caption_encode)[0]
                xt = np.hstack((word, self._global_image_feature.reshape(1, self._hidden_dim)))
                # print('xt', xt.dtype)
                self.xt = np.vstack((self.xt, xt))


            ht, ct, gt, it_act, ft_act = self._lstm_forward(xt, htm1, ctm1, self._lstm_weight_i,
                                                                self._lstm_weight_h, self._lstm_bias)
            st = np.tanh(ct) * sigmoid(np.dot(xt, self._W_x) + np.dot(htm1, self._W_h))
            st_affine = self._relu(np.dot(st, self._Ws_affine) + self._bs_affine)
            st_att = np.dot(st_affine, self._Ws_att) + self._bs_att
            h_affine = np.tanh(np.dot(ht, self._Wh_affine) + self._bh_affine)
            h_att = np.dot(h_affine, self._Wh_att) + self._bh_att
            concate_features = np.vstack(
                (self._image_features, st_affine.reshape((1, self._hidden_dim))))  # self.L + 1, hidden_)dim
            # print('concate_features', concate_features.shape)
            attended_features = np.vstack(
                (self._image_features_proj, st_att.reshape((1, self._hidden_dim))))  # self.L+1 attention_dim
            # print('attended_features', attended_features.shape)
            attention = np.tanh(attended_features + h_att.reshape(1, self._hidden_dim))  # self.L+1 attention_dim
            # print('attention', attention.shape)
            alpha = np.dot(attention, self._alpha)  # self.L+1 ,1
            # print('alpha', alpha.shape)
            att_weights = softmax(alpha, axis=0)
            # print(np.sum(att_weights))
            context = np.sum(att_weights * concate_features, axis=0)
            # print('context', context.shape)
            predict_h = np.tanh(np.dot((context + ht), self._Wcontext_hidden) + self._bcontext_hidden)
            caption_preds = np.dot(predict_h, self._output_weight) + self._output_bias
            caption_preds = softmax(caption_preds)
            # print(caption_preds.dtype)
            caption_encode = np.argmax(caption_preds, axis=1)
            # print(caption_encode)
            self.caption.append(caption_encode[0])
            # print(self.caption)
            self.ht = np.vstack((self.ht, ht.reshape(1, self._hidden_dim)))
            # print(self.ht.dtype)
            self.gt = np.vstack((self.gt, gt.reshape(1, self._hidden_dim)))
            # print(self.gt.dtype)
            self.it_act = np.vstack((self.it_act, it_act.reshape(1, self._hidden_dim)))
            # print(self.it_act.dtype)
            self.ft_act = np.vstack((self.ft_act, ft_act.reshape(1, self._hidden_dim)))
            # print(self.ft_act.dtype)
            self.ct = np.vstack((self.ct, ct.reshape(1, self._hidden_dim)))
            # print(self.ct.dtype)
            if i == 0:
                self.caption_preds = caption_preds
            else:
                self.caption_preds = np.vstack((self.caption_preds, caption_preds))
            if caption_encode[0] == EOS_ENCODED:
                print(self.caption)
                print(self._preprocessor.decode_captions_from_list1d(self.caption))
                return self.caption
        return self.caption

    def _forward_beam_search(self, X, beam_search_captions):
        EOS_ENCODED = self._preprocessor.EOS_TOKEN_LABEL_ENCODED
        SOS_ENCODER = self._preprocessor.SOS_TOKEN_LABEL_ENCODED
        self.caption = beam_search_captions
        sequence_input, img_input = X
        self._img_feature_input = self._image_model.predict(img_input)[0].reshape(self.L, self.D)
        # print('image feature input shape', self._img_feature_input.shape) # 196, 512
        self._image_features_before_act = np.zeros((self.L, self._hidden_dim))
        for i in range(self.L):
            self._image_features_before_act[i] = np.dot(self._img_feature_input[i],
                                                        self._image_features_wieght) + self._image_features_bias
        self._image_features = self._relu(self._image_features_before_act)
        self._average_img_feature = np.mean(self._img_feature_input, axis=0)  # 512,
        # print('average_img_feautre', self._average_img_feature.shape)
        self._global_img_feature_before_act = np.dot(self._average_img_feature,
                                                     self._global_img_feature_weight) + self._global_img_feature_bias
        self._global_img_feature = self._relu(self._global_img_feature_before_act)  # 512,
        # print('global feature shape', self._global_img_feature.shape)
        self._total_static_img_feature = np.dot(self._image_features, self._Wv)  # 196, 512
        self.ht = np.zeros((1, self._hidden_dim), dtype='float32')
        self.ct = np.zeros((1, self._hidden_dim), dtype='float32')
        self.gt = np.zeros((1, self._hidden_dim), dtype='float32')
        self.it_act = np.zeros((1, self._hidden_dim), dtype='float32')
        self.ft_act = np.zeros((1, self._hidden_dim), dtype='float32')
        self.context = np.zeros((1, self._hidden_dim), dtype='float32')
        self.st = np.zeros((1, self._hidden_dim), dtype='float32')
        self.attention = np.zeros((1, self.L), dtype='float32')
        self.beta = np.zeros((1,1), dtype='float32')
        self.c_hat = np.zeros((1, self._hidden_dim), dtype='float32')
        for i in range(len(beam_search_captions)):
            htm1 = self.ht[-1]
            ctm1 = self.ct[-1]
            if i == 0:
                word_embedding = self._embedding.predict(np.array([SOS_ENCODER - 1]))[0]
                xt = np.hstack((word_embedding, self._global_img_feature.reshape(1, self._embedding_dim)))
                self.xt = np.hstack((word_embedding, self._global_img_feature.reshape(1, self._embedding_dim)))
            else:
                word = self._embedding.predict(np.array([beam_search_captions[i-1]-1]))[0]
                xt = np.hstack((word, self._global_img_feature.reshape(1, self._embedding_dim)))
                self.xt = np.vstack((self.xt, xt))
            ht, ct, gt, it_act, ft_act = self._lstm_forward(xt, htm1, ctm1, self._lstm_weight_i,
                                                            self._lstm_weight_h, self._lstm_bias)
            ht_proj = np.dot(ht, self._Wg)
            attention_before_sf = np.dot(np.tanh(ht_proj + self._total_static_img_feature, dtype='float32'), self._V)
            attention = softmax(attention_before_sf, axis=0)
            st = np.tanh(ct) * sigmoid(np.dot(xt, self._Wx) + np.dot(htm1, self._Wh))  # hidden_dim
            zt_extend=np.dot(np.tanh(np.dot(st, self._Ws) + ht_proj), self._V)
            alpha_beta_before_sf = np.concatenate((attention_before_sf, zt_extend), axis=0)
            beta = softmax(alpha_beta_before_sf, axis=0)[-1][0]
            context = np.reshape(np.sum(attention * self._image_features, axis=0), (1, self._hidden_dim))
            c_hat = beta * st + (1 - beta) * context
            ht_context = ht + c_hat  #np.hstack((htm1, context))
            caption_preds = np.dot(ht_context, self._output_weight) + self._output_bias
            self.ht = np.vstack((self.ht, ht.reshape(1, self._hidden_dim)))
            self.gt = np.vstack((self.gt, gt.reshape(1, self._hidden_dim)))
            self.it_act = np.vstack((self.it_act, it_act.reshape(1, self._hidden_dim)))
            self.ft_act = np.vstack((self.ft_act, ft_act.reshape(1, self._hidden_dim)))
            self.ct = np.vstack((self.ct, ct.reshape(1, self._hidden_dim)))
            self.context = np.vstack((self.context, context.reshape(1, self._hidden_dim)))
            self.attention = np.vstack((self.attention, attention.reshape(1, self.L)))
            self.st = np.vstack((self.st, st.reshape(1, self._hidden_dim)))
            self.beta = np.vstack((self.beta, beta.reshape(1,1)))
            self.c_hat = np.vstack((self.c_hat, c_hat))
            if i == 0:
                self.caption_preds = caption_preds
            else:
                self.caption_preds = np.vstack((self.caption_preds, caption_preds))

    def _explain_lstm_single_word(self, t=0):
        '''t: the index of the word to explain start from 1
                    return: relevance of image_encode'''
        if t > len(self.xt):
            raise NotImplementedError('index out of range of captions')
        explain_caption_encode = self.caption[t - 1] - 1  # real caption encode -1 is the encode used for prediciton
        explain_ht = self.ht[t]
        explain_ct = self.ct[t]
        explain_it_act = self.it_act[t]
        explain_gt = self.gt[t]
        explain_context = self.context[t]
        explain_attention = self.attention[t]
        explain_st = self.st[t]
        explain_beta = self.beta[t][0]
        explain_c_hat = self.c_hat[t]
        explain_xht = np.hstack((self.xt[t-1:t], self.ht[t-1:t]))[0]
        explain_relevance = np.zeros((1, len(self.caption_preds[0])))
        explain_relevance[0, explain_caption_encode] = self.caption_preds[t - 1, explain_caption_encode]
        r_V = np.zeros((self.L, self._hidden_dim), dtype='float32')
        r_img_feature_input = np.zeros((self.L, self.D), dtype='float32')
        weight_ig = np.split(self._lstm_weight_i, 4, 1)[2]  # (812, 512)
        weight_hg = np.split(self._lstm_weight_h, 4, 1)[2]  # (512, 512)
        weight_g = np.vstack((weight_ig, weight_hg))  # (600, 300)
        bias_g = np.split(self._lstm_bias, 4)[2]
        self.relevance_rule = self._propagate_relevance_linear_lrp
        r_ht_context = self.relevance_rule(r_in=explain_relevance,
                                           forward_input=explain_ht + explain_c_hat,
                                           forward_output=self.caption_preds[t - 1],
                                           bias=self._output_bias,
                                           bias_nb_units=self._hidden_dim,
                                           weight=self._output_weight)

        r_ht = self.relevance_rule(r_in=r_ht_context,
                                   forward_input=explain_ht,
                                   forward_output=explain_ht + explain_c_hat,
                                   bias=np.zeros(self._hidden_dim),
                                   bias_nb_units=self._hidden_dim,
                                   weight=np.identity(self._hidden_dim))

        r_c_hat = self.relevance_rule(r_in=r_ht_context,
                                      forward_input=explain_c_hat,
                                      forward_output=explain_ht + explain_c_hat,
                                      bias=np.zeros(self._hidden_dim),
                                      bias_nb_units=self._hidden_dim,
                                      weight=np.identity(self._hidden_dim))

        r_context = self.relevance_rule(r_in=r_c_hat,
                                        forward_input=(1 - explain_beta) * explain_context,
                                        forward_output=explain_c_hat,
                                        bias=np.zeros(self._hidden_dim),
                                        bias_nb_units=self._hidden_dim,
                                        weight=np.identity(self._hidden_dim))
        r_st = self.relevance_rule(r_in=r_c_hat,
                                   forward_input=explain_beta * explain_st,
                                   forward_output=explain_c_hat,
                                   bias=np.zeros(self._hidden_dim),
                                   bias_nb_units=self._hidden_dim,
                                   weight=np.identity(self._hidden_dim))
        r_ct = r_ht + r_st
        r_gt = self.relevance_rule(r_in=r_ct,
                                 forward_input=explain_it_act*np.tanh(explain_gt),
                                 forward_output=explain_ct,
                                 bias=np.zeros(self._hidden_dim),
                                 bias_nb_units=self._hidden_dim,
                                 weight=np.identity(self._hidden_dim))
        r_xht = self.relevance_rule(r_in=r_gt,
                                  forward_input=explain_xht,
                                  forward_output=explain_gt,
                                  bias=bias_g,
                                  bias_nb_units=len(explain_xht),
                                  weight=weight_g)
        r_global_img_feature = r_xht[self._embedding_dim:self._embedding_dim + self._embedding_dim]
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
                                                          forward_input=self._image_features[i] * explain_attention[i],
                                                          forward_output=explain_context,
                                                          bias=np.zeros(self._hidden_dim),
                                                          bias_nb_units=self._hidden_dim,
                                                          weight=np.identity(self._hidden_dim))
            r_img_feature_input[i] += self.relevance_rule(r_in=r_V[i],
                                                          forward_input=self._img_feature_input[i],
                                                          forward_output=self._image_features_before_act[i],
                                                          bias=self._image_features_bias,
                                                          bias_nb_units=self.D,
                                                          weight=self._image_features_wieght)
        return r_img_feature_input.reshape(1, int(np.sqrt(self.L)), int(np.sqrt(self.L)), self.D), explain_attention

    def _explain_lstm_single_word_sequence(self, t=0):
        if t > len(self.xt):
            raise NotImplementedError('index out of range of captions')
        explain_caption_encode = self.caption[t - 1] - 1  # real caption encode -1 is the encode used for prediciton
        explain_ht = self.ht[t]
        explain_ct = self.ct[0:t+1]
        explain_it_act = self.it_act[0:t+1]
        explain_ft_act = self.ft_act[0:t+1]
        explain_gt = self.gt[0:t+1]
        explain_context = self.context[t]
        explain_attention = self.attention[t]
        explain_st = self.st[t]
        explain_beta = self.beta[t][0]
        explain_c_hat = self.c_hat[t]
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
        r_ct = np.zeros((t+1, self._hidden_dim))
        r_ht = np.zeros((t+1, self._hidden_dim))
        r_gt = np.zeros((t, self._hidden_dim))
        r_xht = np.zeros((t, self._embedding_dim*2 + self._hidden_dim))

        self.relevance_rule = self._propagate_relevance_linear_lrp

        r_ht_context = self.relevance_rule(r_in=explain_relevance,
                                           forward_input=explain_ht + explain_c_hat,
                                           forward_output=self.caption_preds[t - 1],
                                           bias=self._output_bias,
                                           bias_nb_units=self._hidden_dim,
                                           weight=self._output_weight)

        r_ht[t] = self.relevance_rule(r_in=r_ht_context,
                                      forward_input=explain_ht,
                                      forward_output=explain_ht + explain_c_hat,
                                      bias=np.zeros(self._hidden_dim),
                                      bias_nb_units=self._hidden_dim,
                                      weight=np.identity(self._hidden_dim))

        r_c_hat = self.relevance_rule(r_in=r_ht_context,
                                      forward_input=explain_c_hat,
                                      forward_output=explain_ht + explain_c_hat,
                                      bias=np.zeros(self._hidden_dim),
                                      bias_nb_units=self._hidden_dim,
                                      weight=np.identity(self._hidden_dim))

        r_context = self.relevance_rule(r_in=r_c_hat,
                                        forward_input=(1 - explain_beta) * explain_context,
                                        forward_output=explain_c_hat,
                                        bias=np.zeros(self._hidden_dim),
                                        bias_nb_units=self._hidden_dim,
                                        weight=np.identity(self._hidden_dim))
        r_st = self.relevance_rule(r_in=r_c_hat,
                                   forward_input=explain_beta * explain_st,
                                   forward_output=explain_c_hat,
                                   bias=np.zeros(self._hidden_dim),
                                   bias_nb_units=self._hidden_dim,
                                   weight=np.identity(self._hidden_dim))
        r_ct[t] = r_st

        for i in range(t)[::-1]:
            r_ct[i+1] += r_ht[i+1]
            # print('r_ct', r_ct[i+1])
            r_gt[i] = self.relevance_rule(r_in=r_ct[i+1],
                                       forward_input=explain_it_act[i+1] * np.tanh(explain_gt[i+1]),
                                       forward_output=explain_ct[i+1],
                                       bias=np.zeros(self._hidden_dim),
                                       bias_nb_units=self._hidden_dim,
                                       weight=np.identity(self._hidden_dim))
            # print('r_gt', r_gt[i])
            r_ct[i] = self.relevance_rule(r_in=r_ct[i+1],
                                          forward_input=explain_ft_act[i+1] * explain_ct[i],
                                          forward_output=explain_ct[i+1],
                                          bias=np.zeros(self._hidden_dim),
                                          bias_nb_units=self._hidden_dim,
                                          weight=np.identity(self._hidden_dim))
            r_xht[i] = self.relevance_rule(r_in=r_gt[i],
                                           forward_input=explain_xht[i],
                                           forward_output=explain_gt[i+1],
                                           bias=bias_g,
                                           bias_nb_units=len(explain_xht[0]),
                                           weight=weight_g)
            # print('r_xht', r_xht[i])
            r_ht[i] = r_xht[i][self._embedding_dim*2:]
            # print('r_ht', r_ht[i])
            r_global_img_feature += r_xht[i][self._embedding_dim:self._embedding_dim*2]
            r_wording_embedding[i] = r_xht[i][:self._embedding_dim]
            # print('xt', np.sum(r_xht[i][:self._embedding_dim]))
            # print('r_global_img_feature', r_xht[i][self._embedding_dim:self._embedding_dim*2])
        # print('r_global_img_feature',r_global_img_feature.sum())
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
                                          forward_input=self._image_features[i] * explain_attention[i],
                                          forward_output=explain_context,
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
            self.r_words = self.r_words/max_abso
        self.r_words = self.r_words[1:]
        return r_img_feature_input.reshape(1, int(np.sqrt(self.L)), int(np.sqrt(self.L)), self.D), explain_attention
class ExplainImgCaptioningAdaptiveAttentionGradient(ExplainImgCaptioningAdaptiveAttention):

    def __init__(self,  model, weight_path, dataset_provider, max_caption_length):
        super(ExplainImgCaptioningAdaptiveAttentionGradient, self).__init__(model, weight_path, dataset_provider, max_caption_length)
        self._CNN_explainer = Gradient(self._image_model, neuron_selection_mode="replace")

    def _lstm_forward(self, xt, htm1, ctm1, weight_i, weight_h, bias):
        z = np.dot(xt, weight_i)
        z += np.dot(htm1, weight_h)
        z = z + bias
        z0 = z[:, :self._hidden_dim]
        z1 = z[:, self._hidden_dim: 2 * self._hidden_dim]
        z2 = z[:, 2 * self._hidden_dim: 3 * self._hidden_dim]
        z3 = z[:, 3 * self._hidden_dim:]
        i_act = sigmoid(z0)
        f_act = sigmoid(z1)
        g_act = np.tanh(z2)
        c = f_act * ctm1 + i_act * g_act
        o_act = sigmoid(z3)
        ht = o_act * np.tanh(c)
        ct = c
        return ht, ct, z0, z1, z2, z3, i_act, f_act, g_act, o_act

    def _forward_beam_search(self, X, beam_search_captions):
        SOS_ENCODER = self._preprocessor.SOS_TOKEN_LABEL_ENCODED
        self.caption = beam_search_captions
        sequence_input, img_input = X
        self._img_feature_input = self._image_model.predict(img_input)[0].reshape(self.L, self.D)
        # print('image feature input shape', self._img_feature_input.shape) # 196, 512
        self._image_features_before_act = np.zeros((self.L, self._hidden_dim))
        for i in range(self.L):
            self._image_features_before_act[i] = np.dot(self._img_feature_input[i],
                                                        self._image_features_wieght) + self._image_features_bias
        self._image_features = self._relu(self._image_features_before_act)
        self._average_img_feature = np.mean(self._img_feature_input, axis=0)  # 512,
        # print('average_img_feautre', self._average_img_feature.shape)
        self._global_img_feature_before_act = np.dot(self._average_img_feature,
                                                     self._global_img_feature_weight) + self._global_img_feature_bias
        self._global_img_feature = self._relu(self._global_img_feature_before_act)  # 512,
        # print('global feature shape', self._global_img_feature.shape)
        self._total_static_img_feature = np.dot(self._image_features, self._Wv)  # 196, 512
        for i in range(self.L):
            self._image_features_before_act[i] = np.dot(self._img_feature_input[i],
                                                        self._image_features_wieght) + self._image_features_bias

        self._total_static_img_feature = np.dot(self._image_features, self._Wv)
        self.ht = np.zeros((1, self._hidden_dim), dtype='float32')
        self.ct = np.zeros((1, self._hidden_dim), dtype='float32')
        self.it = np.zeros((1, self._hidden_dim), dtype='float32')
        self.ft = np.zeros((1, self._hidden_dim), dtype='float32')
        self.gt = np.zeros((1, self._hidden_dim), dtype='float32')
        self.ot = np.zeros((1, self._hidden_dim), dtype='float32')
        self.it_act = np.zeros((1, self._hidden_dim), dtype='float32')
        self.ft_act = np.zeros((1, self._hidden_dim), dtype='float32')
        self.gt_act = np.zeros((1, self._hidden_dim), dtype='float32')
        self.ot_act = np.zeros((1, self._hidden_dim), dtype='float32')
        self.context = np.zeros((1, self._hidden_dim), dtype='float32')
        self.st = np.zeros((1, self._hidden_dim), dtype='float32')
        self.attention = np.zeros((1, self.L), dtype='float32')
        self.beta = np.zeros((1, 1), dtype='float32')
        self.c_hat = np.zeros((1, self._hidden_dim), dtype='float32')
        for i in range(len(beam_search_captions)):
            htm1 = self.ht[-1]
            ctm1 = self.ct[-1]
            if i == 0:
                # htm1 = self.ht
                # ctm1 = self.ct
                word_embedding = self._embedding.predict(np.array([SOS_ENCODER - 1]))[0]
                # print('word_embedding', word_embedding.dtype)
                xt = np.hstack((word_embedding, self._global_img_feature.reshape(1, self._embedding_dim)))
                # print('xt', xt.dtype)
                self.xt = np.hstack((word_embedding, self._global_img_feature.reshape(1, self._embedding_dim)))
            else:
                word = self._embedding.predict(np.array([beam_search_captions[i-1]-1]))[0]
                # print(beam_search_captions[i-1]-1)
                xt = np.hstack((word, self._global_img_feature.reshape(1, self._embedding_dim)))
                self.xt = np.vstack((self.xt, xt))
            ht, ct, it, ft, gt, ot, it_act, ft_act, gt_act, ot_act = self._lstm_forward(xt, htm1, ctm1, self._lstm_weight_i,
                                                            self._lstm_weight_h, self._lstm_bias)


            ht_proj = np.dot(ht, self._Wg)
            attention_before_sf = np.dot(np.tanh(ht_proj + self._total_static_img_feature, dtype='float32'), self._V)
            attention = softmax(attention_before_sf, axis=0)
            st = np.tanh(ct) * sigmoid(np.dot(xt, self._Wx) + np.dot(htm1, self._Wh))  # hidden_dim
            zt_extend=np.dot(np.tanh(np.dot(st, self._Ws) + ht_proj), self._V)
            alpha_beta_before_sf = np.concatenate((attention_before_sf, zt_extend), axis=0)
            beta = softmax(alpha_beta_before_sf, axis=0)[-1][0]
            context = np.reshape(np.sum(attention * self._image_features, axis=0), (1, self._hidden_dim))
            c_hat = beta * st + (1 - beta) * context
            ht_context = ht + c_hat  #np.hstack((htm1, context))
            caption_preds = np.dot(ht_context, self._output_weight) + self._output_bias

            self.ht = np.vstack((self.ht, ht.reshape(1, self._hidden_dim)))
            self.ct = np.vstack((self.ct, ct.reshape(1, self._hidden_dim)))
            self.it = np.vstack((self.it, it.reshape(1, self._hidden_dim)))
            self.ft = np.vstack((self.ft, ft.reshape(1, self._hidden_dim)))
            self.gt = np.vstack((self.gt, gt.reshape(1, self._hidden_dim)))
            self.ot = np.vstack((self.ot, ot.reshape(1, self._hidden_dim)))
            self.it_act = np.vstack((self.it_act, it_act.reshape(1, self._hidden_dim)))
            self.ft_act = np.vstack((self.ft_act, ft_act.reshape(1, self._hidden_dim)))
            self.gt_act = np.vstack((self.gt_act, gt_act.reshape(1, self._hidden_dim)))
            self.ot_act = np.vstack((self.ot_act, ot_act.reshape(1, self._hidden_dim)))
            self.context = np.vstack((self.context, context.reshape(1, self._hidden_dim)))
            self.attention = np.vstack((self.attention, attention.reshape(1, self.L)))
            self.st = np.vstack((self.st, st.reshape(1, self._hidden_dim)))
            self.beta = np.vstack((self.beta, beta.reshape(1, 1)))
            self.c_hat = np.vstack((self.c_hat, c_hat))
            if i == 0:
                self.caption_preds = caption_preds
            else:
                self.caption_preds = np.vstack((self.caption_preds, caption_preds))

    def _lstm_decoder_backward(self, t):

        '''During the forward pass, we get the ht, ct, it, ft, gt, ot and their activations, beta, attention, context, c_hat, captionpreds'''
        explain_caption_encode = self.caption[t - 1] - 1
        d_caption_preds = np.zeros((1, len(self.caption_preds[0])))
        d_caption_preds[0, explain_caption_encode] = 1.0

        d_V = np.zeros((self.L, self._hidden_dim), dtype='float32')
        d_it_act = np.zeros((t, self._hidden_dim), dtype='float32')
        d_ft_act = np.zeros((t, self._hidden_dim), dtype='float32')
        d_gt_act = np.zeros((t, self._hidden_dim), dtype='float32')
        d_ot_act = np.zeros((t, self._hidden_dim), dtype='float32')
        d_it = np.zeros((t, self._hidden_dim), dtype='float32')
        d_ft = np.zeros((t, self._hidden_dim), dtype='float32')
        d_gt = np.zeros((t, self._hidden_dim), dtype='float32')
        d_ot = np.zeros((t, self._hidden_dim), dtype='float32')
        d_ht = np.zeros((t+1, self._hidden_dim), dtype='float32')
        d_ct = np.zeros((t+1, self._hidden_dim), dtype='float32')
        d_xt = np.zeros((t, self._embedding_dim+self._hidden_dim), dtype='float32')
        d_words = np.zeros((t, self._embedding_dim))
        d_img_feature_input = np.zeros((self.L, self.D), dtype='float32')
        d_ht_context= np.dot(d_caption_preds, self._output_weight.transpose()) # (1, hidden_dim)
        '''trivial'''
        d_c_hat = d_ht_context * 1
        d_context = d_c_hat
        d_ht[t] = d_ht_context * 1
        d_global_img_feature = np.zeros(self._hidden_dim)
        for i in range(self.L):
            d_V[i] = d_context * self.attention[t, i]
        d_V[self._image_features<=0] = 0
        for i in range(t)[::-1]:
            d_ot_act[i] = d_ht[i+1] * np.tanh(self.ct[i+1])
            d_ct[i+1] += d_ht[i+1] * self.ot_act[i+1] * (1. - (np.tanh(self.ct[i+1]))**2)
            d_ft_act[i] = d_ct[i+1] * self.ct[i]
            d_ct[i] = d_ct[i+1] * self.ft_act[i+1]
            d_it_act[i] = d_ct[i+1] * self.gt_act[i+1]
            d_gt_act[i] = d_ct[i+1] * self.it_act[i+1]
            d_it[i] = d_it_act[i] * self.it_act[i + 1] * (1 - self.it_act[i + 1])
            d_ft[i] = d_ft_act[i] * self.ft_act[i + 1] * (1 - self.ft_act[i + 1])
            d_ot[i] = d_ot_act[i] * self.ot_act[i + 1] * (1 - self.ot_act[i + 1])
            d_gt[i] = d_gt_act[i] * (1 - (self.gt_act[i + 1]) ** 2)
            d_gates = np.hstack((d_it[i: i+1], d_ft[i: i+1], d_gt[i: i+1], d_ot[i: i+1]))
            d_ht[i] = np.dot(d_gates, self._lstm_weight_h.transpose())
            d_xt[i] = np.dot(d_gates, self._lstm_weight_i.transpose())
            d_global_img_feature += d_xt[i][self._embedding_dim: ]
            d_words[i] = d_xt[i][:self._embedding_dim]
        d_global_img_feature[self._global_img_feature[0]<=0] = 0
        d_average_img_feature = np.dot(d_global_img_feature, self._global_img_feature_weight.transpose())
        for i in range(self.L):
            d_img_feature_input[i] = 1.0 * d_average_img_feature / self.L
            d_img_feature_input[i] += np.dot(d_V[i], self._image_features_wieght.transpose())
        self.r_words = np.sum(d_words, axis=-1)
        return d_img_feature_input.reshape(1, int(np.sqrt(self.L)), int(np.sqrt(self.L)), self.D)

    def _explain_sentence(self, ):
        image_feature_relevance = []
        for i in range(len(self.caption)-1):
            relevance = self._lstm_decoder_backward(i+1)
            image_feature_relevance.append(relevance)
        return image_feature_relevance

    def process_beam_search(self, beam_size=3, ):
        print('Gradient')
        data_generator = self._dataset_provider.test_set(include_datum=True)
        count = 0
        for X, y, data in data_generator:
            imgs_path = map(attrgetter('img_path'), data)
            imgs_path = list(imgs_path)
            count += 1
            if count ==1:  # 9
                beam_search_captions_encoded = self._beam_search(X, beam_size)[0]
                beam_search_captions = self._preprocessor.decode_captions_from_list1d(beam_search_captions_encoded)
                img_filename = imgs_path[0].split('/')[-1]
                save_folder = os.path.join(self._weight_path + 'explanation', img_filename.split('.')[0])
                if not os.path.isdir(save_folder):
                    os.makedirs(save_folder)
                img_original = Image.open(imgs_path[0])
                img_original = img_original.resize((224, 224))
                img_original.save(os.path.join(save_folder, img_filename))
                sequence_input, img_input, = X
                self._forward_beam_search(X, beam_search_captions_encoded)
                img_encode_relevance= self._explain_sentence()
                x = int(np.sqrt(len(img_encode_relevance)))
                y = int(np.ceil(len(img_encode_relevance) / x))
                _, axes = plt.subplots(y, x, sharex="col", sharey="row", figsize=(20,20))
                # plt.axis('off')
                axes = axes.flatten()
                for i in range(len(img_encode_relevance)):
                    relevance = self._explain_CNN(img_input, img_encode_relevance[i])
                    # print(beam_search_captions[0].split()[i])
                    # print(np.sum(relevance[0] == 0))
                    # print(np.sum(relevance[0] > 0))
                    # print(np.sum(relevance[0] < 0))
                    channels_first = K.image_data_format() == "channels_first"
                    hp = postprocess(relevance, None, channels_first)
                    hp = heatmap(hp)[0]
                    axes[i].set_title(self._preprocessor._word_of[beam_search_captions_encoded[i]], fontsize=18)
                    axes[i].imshow(hp)
                plt.savefig(os.path.join(save_folder, img_filename.split('.')[0] + 'gradient_hm.jpg'))
                break
class ExplainImgCaptioningAdaptiveAttentionInputTimesGradient(ExplainImgCaptioningAdaptiveAttentionGradient):
    def __init__(self,  model, weight_path, dataset_provider, max_caption_length):
        super(ExplainImgCaptioningAdaptiveAttentionGradient, self).__init__(model, weight_path, dataset_provider, max_caption_length)
        self._CNN_explainer = InputTimesGradient(self._image_model, neuron_selection_mode="replace")

    def process_beam_search(self, beam_size=3, ):
        print('InputTimesGradient')
        data_generator = self._dataset_provider.test_set(include_datum=True)
        count = 0
        for X, y, data in data_generator:
            imgs_path = map(attrgetter('img_path'), data)
            imgs_path = list(imgs_path)
            count += 1
            if count ==1:  # 9
                beam_search_captions_encoded = self._beam_search(X, beam_size)[0]
                beam_search_captions = self._preprocessor.decode_captions_from_list1d(beam_search_captions_encoded)
                img_filename = imgs_path[0].split('/')[-1]
                save_folder = os.path.join(self._weight_path + 'explanation', img_filename.split('.')[0])
                if not os.path.isdir(save_folder):
                    os.makedirs(save_folder)
                img_original = Image.open(imgs_path[0])
                img_original = img_original.resize((224, 224))
                img_original.save(os.path.join(save_folder, img_filename))
                sequence_input, img_input, = X
                self._forward_beam_search(X, beam_search_captions_encoded)
                img_encode_relevance= self._explain_sentence()
                x = int(np.sqrt(len(img_encode_relevance)))
                y = int(np.ceil(len(img_encode_relevance) / x))
                _, axes = plt.subplots(y, x, sharex="col", sharey="row", figsize=(20,20))
                # plt.axis('off')
                axes = axes.flatten()
                for i in range(len(img_encode_relevance)):
                    relevance = self._explain_CNN(img_input, img_encode_relevance[i])
                    # print(beam_search_captions[0].split()[i])
                    # print(np.sum(relevance[0] == 0))
                    # print(np.sum(relevance[0] > 0))
                    # print(np.sum(relevance[0] < 0))
                    channels_first = K.image_data_format() == "channels_first"
                    color_conversion = None
                    hp = postprocess(relevance, color_conversion, channels_first)
                    hp = heatmap(hp)[0]
                    axes[i].set_title(self._preprocessor._word_of[beam_search_captions_encoded[i]], fontsize=18)
                    axes[i].imshow(hp)
                plt.savefig(os.path.join(save_folder, img_filename.split('.')[0] + 'gradienttimesinput_hm.jpg'))
                break
class ExplainImgCaptioningAdaptiveAttentionGuidedGradcam(ExplainImgCaptioningAdaptiveAttentionGradient):
    def __init__(self,  model, weight_path, dataset_provider, max_caption_length=20):
        super(ExplainImgCaptioningAdaptiveAttentionGradient, self).__init__(model, weight_path, dataset_provider, max_caption_length)
        self._CNN_explainer= GuidedBackprop(self._image_model, neuron_selection_mode="replace")

    def _explain_CNN(self, X, relevance_value):
        gradcamp = self.grad_cam(self._img_feature_input, relevance_value[0])
        guided_backprop_relevance = self._CNN_explainer.analyze([X, relevance_value])
        # if you use guided gradcam use the guided_backprop line
        relevance = guided_backprop_relevance[0] * gradcamp[..., np.newaxis]
        # if explain gradcam, use the np.ones line
        # relevance = np.ones(guided_backprop_relevance.shape)[0] * gradcamp[..., np.newaxis]
        return relevance[np.newaxis,:]

    def grad_cam(self, img_feature, grads):
        weights = np.mean(grads, axis=(0, 1))
        conv_output = img_feature.reshape((int(np.sqrt(self.L)), int(np.sqrt(self.L)), self.D))
        cam = np.zeros(conv_output.shape[0:2], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * conv_output[:, :, i]
        cam = skimage.transform.pyramid_expand(cam, upscale=16, sigma=20,
                                               multichannel=False)
        cam = np.maximum(cam, 0)
        cam_heatmap = cam/(np.max(np.abs(cam)) + 1e-6)
        return cam_heatmap

    def process_beam_search(self, beam_size=3):
        print('Guided_gradcam')
        data_generator = self._dataset_provider.test_set(include_datum=True)
        count = 0
        for X, y, data in data_generator:
            imgs_path = map(attrgetter('img_path'), data)
            imgs_path = list(imgs_path)
            count += 1
            if count ==1:  # 9
                beam_search_captions_encoded = self._beam_search(X, beam_size)[0]
                beam_search_captions = self._preprocessor.decode_captions_from_list1d(beam_search_captions_encoded)
                img_filename = imgs_path[0].split('/')[-1]
                save_folder = os.path.join(self._weight_path + 'explanation', img_filename.split('.')[0])
                if not os.path.isdir(save_folder):
                    os.makedirs(save_folder)
                img_original = Image.open(imgs_path[0])
                img_original = img_original.resize((224, 224))
                img_original.save(os.path.join(save_folder, img_filename))
                sequence_input, img_input, = X
                self._forward_beam_search(X, beam_search_captions_encoded)
                '''for a full sentence'''
                img_encode_relevance= self._explain_sentence()
                x = int(np.sqrt(len(img_encode_relevance)))
                y = int(np.ceil(len(img_encode_relevance) / x))
                _, axes = plt.subplots(y, x, sharex="col", sharey="row", figsize=(20,20))
                # plt.axis('off')
                axes = axes.flatten()
                for i in range(len(img_encode_relevance)):
                    relevance = self._explain_CNN(img_input, img_encode_relevance[i])
                    # print(beam_search_captions[0].split()[i])
                    # print(np.sum(relevance[0] == 0))
                    # print(np.sum(relevance[0] > 0))
                    # print(np.sum(relevance[0] < 0))
                    channels_first = K.image_data_format() == "channels_first"
                    color_conversion = "BGRtoRGB"
                    hp = postprocess(relevance, color_conversion, channels_first)
                    hp = heatmap(hp)[0]
                    axes[i].set_title(self._preprocessor._word_of[beam_search_captions_encoded[i]], fontsize=18)
                    axes[i].imshow(hp)
                plt.savefig(os.path.join(save_folder, img_filename.split('.')[0] + 'guidedgradcam_hm.jpg'))
                break



class ExplainImgCaptioningGridTDModel(ExplainImgCaptioningAttentionModel):

    def __init__(self,model, weight_path, dataset_provider, max_caption_length=20):
        super(ExplainImgCaptioningGridTDModel, self).__init__(model, weight_path, dataset_provider, max_caption_length)

        self._global_img_feature_weight_bm, self._global_img_feature_bias_bm = self._keras_model.get_layer(
            'global_img_feature').get_weights()

        self._image_features_weight_bm, self._image_features_bias_bm = self._keras_model.get_layer(
            'image_features').get_weights()
        self._embedding_bm = Model(self._keras_model.get_layer('embedding_1').input,
                                self._keras_model.get_layer('embedding_1').output)
        self._attention_layer_bm = self._keras_model.get_layer('external_bottom_up_attention_adaptive_1')
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

    def _forward_single(self, X):
        EOS_ENCODED = self._preprocessor.EOS_TOKEN_LABEL_ENCODED
        SOS_ENCODER = self._preprocessor.SOS_TOKEN_LABEL_ENCODED
        sequence_input, img_input = X
        self._image_features_input_bm  = self._image_model.predict(img_input)[0].reshape(self.L, self.D)
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
        self.x2t = np.zeros((1, self._hidden_dim *2))
        self.x1t = np.zeros((1, self._hidden_dim + self._embedding_dim + self._embedding_dim))
        self.caption = []

        for i in range(self._max_caption_length):
            h1tm1 = self.h1t[-1].reshape(1, self._hidden_dim)
            c1tm1 = self.c1t[-1].reshape(1, self._hidden_dim)
            h2tm1 = self.h2t[-1].reshape(1, self._hidden_dim)
            c2tm1 = self.c2t[-1].reshape(1, self._hidden_dim)
            if i == 0:
                word_embedding = self._embedding_bm.predict(np.array([SOS_ENCODER-1]))[0]
                x1t = np.hstack((h2tm1, self._global_image_feature_bm.reshape(1, self._embedding_dim), word_embedding))
            else:
                word_embedding = self._embedding_bm.predict(caption_encode)[0]
                x1t = np.hstack((h2tm1, self._global_image_feature_bm, word_embedding))
            h1t, c1t, g1t, i1t_act, f1t_act = self._lstm_forward(x1t, h1tm1, c1tm1, self._top_down_lstm_weight_i,
                                                            self._top_down_lstm_weight_h, self._top_down_lstm_weight_bias)
            h_proj = np.dot(h1t, self._W_ha)
            attention_bf_act = np.dot(np.tanh(self._image_features_proj_bm + h_proj), self._W_a)
            attention = softmax(attention_bf_act, axis=0)
            context = np.reshape(np.sum(attention * self._image_features_bm, axis=0), (1, self._hidden_dim))
            st = np.tanh(c1t) * sigmoid(np.dot(x1t, self._W_x) + np.dot(h1tm1, self._W_h))
            zt_extend = np.dot(np.tanh(np.dot(st, self._W_s) + h_proj), self._W_a)
            alpha_beta_before_sf = np.concatenate((attention_bf_act, zt_extend), axis=0)
            beta = softmax(alpha_beta_before_sf, axis=0)[-1][0]
            context_hat = beta * st + (1-beta) * context

            x2t = np.hstack((context_hat, h1t.reshape(1, self._hidden_dim)))
            h2t, c2t, g2t, i2t_act, f2t_act = self._lstm_forward(x2t, h2tm1, c2tm1, self._language_lstm_weight_i,
                                                                 self._language_lstm_weight_h, self._language_lstm_bias)
            caption_preds = np.dot(h2t + context_hat, self._output_weight_bm) + self._output_bias_bm
            caption_encode = np.argmax(caption_preds, axis=1)
            self.caption.append(caption_encode[0] + 1)
            self.h1t = np.vstack((self.h1t, h1t.reshape(1, self._hidden_dim)))
            self.h2t = np.vstack((self.h2t, h2t.reshape(1, self._hidden_dim)))
            self.c1t = np.vstack((self.c1t, c1t.reshape(1, self._hidden_dim)))
            self.c2t = np.vstack((self.c2t, c2t.reshape(1, self._hidden_dim)))
            self.x1t = np.vstack((self.x1t, x1t))
            self.x2t = np.vstack((self.x2t, x2t))

            if i == 0:
                self.caption_preds = caption_preds
            else:
                self.caption_preds = np.vstack((self.caption_preds, caption_preds))
            if caption_encode[0]+1 == EOS_ENCODED:
                print(self.caption)
                print(self._preprocessor.decode_captions_from_list1d(self.caption))
                return self.caption
        return self.caption

    def _forward_beam_search(self, X, beam_search_captions):
        EOS_ENCODED = self._preprocessor.EOS_TOKEN_LABEL_ENCODED
        SOS_ENCODER = self._preprocessor.SOS_TOKEN_LABEL_ENCODED
        sequence_input, img_input = X
        self.caption = beam_search_captions
        self._image_features_input_bm  = self._image_model.predict(img_input)[0].reshape(self.L, self.D)
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
                x1t = np.hstack((h2tm1, self._global_image_feature_bm.reshape(1, self._embedding_dim), word_embedding))
            else:
                word_embedding = self._embedding_bm.predict(np.array([beam_search_captions[i-1]-1]))[0]
                x1t = np.hstack((h2tm1, self._global_image_feature_bm.reshape(1, self._embedding_dim), word_embedding))
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
class ExplainImgCaptioningGridTDGradient(ExplainImgCaptioningGridTDModel):
    def __init__(self, model, weight_path, dataset_provider, max_caption_length):
        super(ExplainImgCaptioningGridTDGradient, self).__init__(model, weight_path, dataset_provider, max_caption_length)
        self._CNN_explainer = Gradient(self._image_model, neuron_selection_mode="replace")

    def _lstm_forward(self, xt, htm1, ctm1, weight_i, weight_h, bias):
        z = np.dot(xt, weight_i)
        z += np.dot(htm1, weight_h)
        z = z + bias
        z0 = z[:, :self._hidden_dim]
        z1 = z[:, self._hidden_dim: 2 * self._hidden_dim]
        z2 = z[:, 2 * self._hidden_dim: 3 * self._hidden_dim]
        z3 = z[:, 3 * self._hidden_dim:]
        i_act = sigmoid(z0)
        f_act = sigmoid(z1)
        g_act = np.tanh(z2)
        c = f_act * ctm1 + i_act * g_act
        o_act = sigmoid(z3)
        ht = o_act * np.tanh(c)
        ct = c
        return ht, ct, z0, z1, z2, z3, i_act, f_act, g_act, o_act

    def _forward_beam_search(self, X, beam_search_captions):
        EOS_ENCODED = self._preprocessor.EOS_TOKEN_LABEL_ENCODED
        SOS_ENCODER = self._preprocessor.SOS_TOKEN_LABEL_ENCODED
        sequence_input, img_input = X
        self.caption = beam_search_captions
        self._image_features_input_bm  = self._image_model.predict(img_input)[0].reshape(self.L, self.D)
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
        self.i1t = np.zeros((1, self._hidden_dim), dtype='float32')
        self.f1t = np.zeros((1, self._hidden_dim), dtype='float32')
        self.g1t = np.zeros((1, self._hidden_dim), dtype='float32')
        self.o1t = np.zeros((1, self._hidden_dim), dtype='float32')
        self.i2t = np.zeros((1, self._hidden_dim), dtype='float32')
        self.f2t = np.zeros((1, self._hidden_dim), dtype='float32')
        self.g2t = np.zeros((1, self._hidden_dim), dtype='float32')
        self.o2t = np.zeros((1, self._hidden_dim), dtype='float32')
        self.i1t_act = np.zeros((1, self._hidden_dim), dtype='float32')
        self.f1t_act = np.zeros((1, self._hidden_dim), dtype='float32')
        self.g1t_act = np.zeros((1, self._hidden_dim), dtype='float32')
        self.o1t_act = np.zeros((1, self._hidden_dim), dtype='float32')
        self.i2t_act = np.zeros((1, self._hidden_dim), dtype='float32')
        self.f2t_act = np.zeros((1, self._hidden_dim), dtype='float32')
        self.g2t_act = np.zeros((1, self._hidden_dim), dtype='float32')
        self.o2t_act = np.zeros((1, self._hidden_dim), dtype='float32')
        self.x2t = np.zeros((1, self._hidden_dim + self._hidden_dim))
        self.x1t = np.zeros((1, self._hidden_dim + self._embedding_dim + self._embedding_dim))
        self.context = np.zeros((1, self._hidden_dim))
        self.attention = np.zeros((1, self.L))
        self.st = np.zeros((1, self._hidden_dim))
        self.beta = np.zeros((1,1))
        self.context_hat = np.zeros((1, self._hidden_dim))
        for i in range(len(beam_search_captions)):
            h1tm1 = self.h1t[-1].reshape(1, self._hidden_dim)
            c1tm1 = self.c1t[-1].reshape(1, self._hidden_dim)
            h2tm1 = self.h2t[-1].reshape(1, self._hidden_dim)
            c2tm1 = self.c2t[-1].reshape(1, self._hidden_dim)
            if i == 0:
                word_embedding = self._embedding_bm.predict(np.array([SOS_ENCODER - 1]))[0]
                x1t = np.hstack((h2tm1, self._global_image_feature_bm.reshape(1, self._embedding_dim), word_embedding))
            else:
                word_embedding = self._embedding_bm.predict(np.array([beam_search_captions[i-1]-1]))[0]
                x1t = np.hstack((h2tm1, self._global_image_feature_bm.reshape(1, self._embedding_dim), word_embedding))
            h1t, c1t, i1t, f1t, g1t, o1t, i1t_act, f1t_act, g1t_act, o1t_act = self._lstm_forward(x1t, h1tm1, c1tm1, self._top_down_lstm_weight_i,
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
            h2t, c2t, i2t, f2t, g2t, o2t, i2t_act, f2t_act, g2t_act, o2t_act = self._lstm_forward(x2t, h2tm1, c2tm1, self._language_lstm_weight_i,
                                                                 self._language_lstm_weight_h, self._language_lstm_bias)
            caption_preds = np.dot(h2t, self._output_weight_bm) + self._output_bias_bm
            self.h1t = np.vstack((self.h1t, h1t.reshape(1, self._hidden_dim)))
            self.h2t = np.vstack((self.h2t, h2t.reshape(1, self._hidden_dim)))
            self.c1t = np.vstack((self.c1t, c1t.reshape(1, self._hidden_dim)))
            self.c2t = np.vstack((self.c2t, c2t.reshape(1, self._hidden_dim)))
            self.x1t = np.vstack((self.x1t, x1t))
            self.x2t = np.vstack((self.x2t, x2t))
            self.i1t = np.vstack((self.i1t, i1t.reshape(1, self._hidden_dim)))
            self.f1t = np.vstack((self.f1t, f1t.reshape(1, self._hidden_dim)))
            self.g1t = np.vstack((self.g1t, g1t.reshape(1, self._hidden_dim)))
            self.o1t = np.vstack((self.o1t, o1t.reshape(1, self._hidden_dim)))
            self.i2t = np.vstack((self.i2t, i2t.reshape(1, self._hidden_dim)))
            self.f2t = np.vstack((self.f2t, f2t.reshape(1, self._hidden_dim)))
            self.g2t = np.vstack((self.g2t, g2t.reshape(1, self._hidden_dim)))
            self.o2t = np.vstack((self.o2t, o2t.reshape(1, self._hidden_dim)))
            self.i1t_act = np.vstack((self.i1t_act, i1t_act.reshape(1, self._hidden_dim)))
            self.f1t_act = np.vstack((self.f1t_act, f1t_act.reshape(1, self._hidden_dim)))
            self.g1t_act = np.vstack((self.g1t_act, g1t_act.reshape(1, self._hidden_dim)))
            self.o1t_act = np.vstack((self.o1t_act, o1t_act.reshape(1, self._hidden_dim)))
            self.i2t_act = np.vstack((self.i2t_act, i2t_act.reshape(1, self._hidden_dim)))
            self.f2t_act = np.vstack((self.f2t_act, f2t_act.reshape(1, self._hidden_dim)))
            self.g2t_act = np.vstack((self.g2t_act, g2t_act.reshape(1, self._hidden_dim)))
            self.o2t_act = np.vstack((self.o2t_act, o2t_act.reshape(1, self._hidden_dim)))
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

    def _lstm_decoder_backward(self, t):
        explain_caption_encode = self.caption[t - 1] - 1
        d_caption_preds = np.zeros((1, len(self.caption_preds[0])))
        d_caption_preds[0, explain_caption_encode] = 1.0

        d_i1t_act = np.zeros((t, self._hidden_dim), dtype='float32')
        d_f1t_act = np.zeros((t, self._hidden_dim), dtype='float32')
        d_g1t_act = np.zeros((t, self._hidden_dim), dtype='float32')
        d_o1t_act = np.zeros((t, self._hidden_dim), dtype='float32')
        d_i1t = np.zeros((t, self._hidden_dim), dtype='float32')
        d_f1t = np.zeros((t, self._hidden_dim), dtype='float32')
        d_g1t = np.zeros((t, self._hidden_dim), dtype='float32')
        d_o1t = np.zeros((t, self._hidden_dim), dtype='float32')
        d_h1t = np.zeros((t + 1, self._hidden_dim), dtype='float32')
        d_c1t = np.zeros((t + 1, self._hidden_dim), dtype='float32')
        d_i2t_act = np.zeros((t, self._hidden_dim), dtype='float32')
        d_f2t_act = np.zeros((t, self._hidden_dim), dtype='float32')
        d_g2t_act = np.zeros((t, self._hidden_dim), dtype='float32')
        d_o2t_act = np.zeros((t, self._hidden_dim), dtype='float32')
        d_i2t = np.zeros((t, self._hidden_dim), dtype='float32')
        d_f2t = np.zeros((t, self._hidden_dim), dtype='float32')
        d_g2t = np.zeros((t, self._hidden_dim), dtype='float32')
        d_o2t = np.zeros((t, self._hidden_dim), dtype='float32')
        d_h2t = np.zeros((t + 1, self._hidden_dim), dtype='float32')
        d_c2t = np.zeros((t + 1, self._hidden_dim), dtype='float32')
        d_xt1 = np.zeros((t, self._embedding_dim * 2 + self._hidden_dim))
        d_xt2 = np.zeros((t, self._hidden_dim * 2))
        d_V = np.zeros((self.L, self._hidden_dim))
        d_words = np.zeros((t, self._embedding_dim))
        d_img_feature_input = np.zeros((self.L, self.D), dtype='float32')
        d_global_image_feature = np.zeros((1, self._embedding_dim))
        d_context_hat = np.zeros((t, self._hidden_dim))

        d_h2t_context_for_predict = np.dot(d_caption_preds, self._output_weight_bm.transpose())
        d_context_hat[t-1] = d_h2t_context_for_predict
        d_h2t[t] = d_h2t_context_for_predict

        for i in range(t)[::-1]:
            d_o2t_act[i] = d_h2t[i + 1] * np.tanh(self.c2t[i + 1])
            d_c2t[i + 1] += d_h2t[i + 1] * self.o2t_act[i + 1] * (1. - (np.tanh(self.c2t[i + 1])) ** 2)
            d_f2t_act[i] = d_c2t[i + 1] * self.c2t[i]
            d_c2t[i] = d_c2t[i+1] * self.f2t_act[i+1]
            d_i2t_act[i] = d_c2t[i+1] * self.g2t_act[i+1]
            d_g2t_act[i] = d_c2t[i+1] * self.i2t_act[i+1]
            d_i2t[i] = d_i2t_act[i] * self.i2t_act[i + 1] * (1 - self.i2t_act[i + 1])
            d_f2t[i] = d_f2t_act[i] * self.f2t_act[i + 1] * (1 - self.f2t_act[i + 1])
            d_o2t[i] = d_o2t_act[i] * self.o2t_act[i + 1] * (1 - self.o2t_act[i + 1])
            d_g2t[i] = d_g2t_act[i] * (1 - (self.g2t_act[i + 1]) ** 2)
            d_gates2 = np.hstack((d_i2t[i: i+1], d_f2t[i: i+1], d_g2t[i: i+1], d_o2t[i: i+1]))
            d_h2t[i] = np.dot(d_gates2, self._language_lstm_weight_h.transpose())
            d_xt2[i] = np.dot(d_gates2, self._language_lstm_weight_i.transpose())
            d_context_hat[i] += d_xt2[i][: self._hidden_dim]
            d_context = d_context_hat[i] * (1 - self.beta[i+1][0])
            d_h1t[i+1] += d_xt2[i][self._hidden_dim:]
            d_o1t_act[i] = d_h1t[i + 1] * np.tanh(self.c1t[i + 1])
            d_c1t[i + 1] += d_h1t[i + 1] * self.o1t_act[i + 1] * (1. - (np.tanh(self.c1t[i + 1])) ** 2)
            d_f1t_act[i] = d_c1t[i + 1] * self.c1t[i]
            d_c1t[i] = d_c1t[i+1] * self.f1t_act[i+1]
            d_i1t_act[i] = d_c1t[i+1] * self.g1t_act[i+1]
            d_g1t_act[i] = d_c1t[i+1] * self.i1t_act[i+1]
            d_i1t[i] = d_i1t_act[i] * self.i1t_act[i + 1] * (1 - self.i1t_act[i + 1])
            d_f1t[i] = d_f1t_act[i] * self.f1t_act[i + 1] * (1 - self.f1t_act[i + 1])
            d_o1t[i] = d_o1t_act[i] * self.o1t_act[i + 1] * (1 - self.o1t_act[i + 1])
            d_g1t[i] = d_g1t_act[i] * (1 - (self.g1t_act[i + 1]) ** 2)
            d_gates1 = np.hstack((d_i1t[i: i+1], d_f1t[i: i+1], d_g1t[i: i+1], d_o1t[i: i+1]))
            d_h1t[i] = np.dot(d_gates1, self._top_down_lstm_weight_h.transpose())
            d_xt1[i] = np.dot(d_gates1, self._top_down_lstm_weight_i.transpose())
            d_global_image_feature += d_xt1[i][self._hidden_dim:self._hidden_dim+self._embedding_dim]
            d_words[i] = d_xt1[i][self._hidden_dim+self._embedding_dim:]
            for k in range(self.L):
                d_V[k] += d_context * self.attention[i+1][k]
            d_h2t[i] += d_xt1[i][:self._hidden_dim]
        d_global_image_feature[0][self._global_image_feature_bm<=0] = 0
        d_average_image_feature = np.dot(d_global_image_feature, self._global_img_feature_weight_bm.transpose())
        d_V[self._image_features_bm<=0] = 0
        self.r_words = np.sum(d_words, axis=-1)
        for k in range(self.L):
            d_img_feature_input[k] = np.dot(d_V[k], self._image_features_weight_bm.transpose())
            d_img_feature_input[k] += d_average_image_feature[0] / self.L

        return d_img_feature_input.reshape(1, int(np.sqrt(self.L)), int(np.sqrt(self.L)), self.D)

    def _explain_sentence(self):
        image_feature_relevance = []
        for i in range(len(self.caption)-1):
            relevance = self._lstm_decoder_backward(i+1)
            image_feature_relevance.append(relevance)
        return image_feature_relevance

    def process_beam_search(self, beam_size=3):
        print('Gradient')
        data_generator = self._dataset_provider.test_set(include_datum=True)
        count = 0
        for X, y, data in data_generator:
            imgs_path = map(attrgetter('img_path'), data)
            imgs_path = list(imgs_path)
            count += 1
            if count ==1:  # 9
                beam_search_captions_encoded = self._beam_search(X, beam_size)[0]
                beam_search_captions = self._preprocessor.decode_captions_from_list1d(beam_search_captions_encoded)
                img_filename = imgs_path[0].split('/')[-1]
                save_folder = os.path.join(self._weight_path + 'explanation', img_filename.split('.')[0])
                if not os.path.isdir(save_folder):
                    os.makedirs(save_folder)
                img_original = Image.open(imgs_path[0])
                img_original = img_original.resize((224, 224))
                img_original.save(os.path.join(save_folder, img_filename))
                sequence_input, img_input, = X
                self._forward_beam_search(X, beam_search_captions_encoded)
                '''for a full sentence'''
                img_encode_relevance= self._explain_sentence()
                x = int(np.sqrt(len(img_encode_relevance)))
                y = int(np.ceil(len(img_encode_relevance) / x))
                _, axes = plt.subplots(y, x, sharex="col", sharey="row", figsize=(20,20))
                # plt.axis('off')
                axes = axes.flatten()
                for i in range(len(img_encode_relevance)):
                    relevance = self._explain_CNN(img_input, img_encode_relevance[i])
                    # print(beam_search_captions[0].split()[i])
                    # print(np.sum(relevance[0] == 0))
                    # print(np.sum(relevance[0] > 0))
                    # print(np.sum(relevance[0] < 0))
                    channels_first = K.image_data_format() == "channels_first"
                    color_conversion = "BGRtoRGB" if "BGR" == "BGR" else None
                    hp = postprocess(relevance, color_conversion, channels_first)
                    hp = heatmap(hp)[0]
                    axes[i].set_title(self._preprocessor._word_of[beam_search_captions_encoded[i]], fontsize=18)
                    axes[i].imshow(hp)
                plt.show()
                plt.savefig(os.path.join(save_folder, img_filename.split('.')[0] + 'gradient_hm.jpg'))

                break
class ExplainImgCaptioningGridTDGradientTimesInput(ExplainImgCaptioningGridTDGradient):
    def __init__(self, model, weight_path, dataset_provider, max_caption_length):
        super(ExplainImgCaptioningGridTDGradient, self).__init__(model, weight_path, dataset_provider, max_caption_length)
        self._CNN_explainer = InputTimesGradient(self._image_model, neuron_selection_mode="replace")

    def process_beam_search(self, beam_size=3, ):
        print('InputTimesGradient')
        data_generator = self._dataset_provider.test_set(include_datum=True)
        count = 0
        for X, y, data in data_generator:
            imgs_path = map(attrgetter('img_path'), data)
            imgs_path = list(imgs_path)
            count += 1
            if count ==2:  # 9
                beam_search_captions_encoded = self._beam_search(X, beam_size)[0]
                beam_search_captions = self._preprocessor.decode_captions_from_list1d(beam_search_captions_encoded)
                img_filename = imgs_path[0].split('/')[-1]
                save_folder = os.path.join(self._weight_path + 'explanation', img_filename.split('.')[0])
                if not os.path.isdir(save_folder):
                    os.makedirs(save_folder)
                img_original = Image.open(imgs_path[0])
                img_original = img_original.resize((224, 224))
                img_original.save(os.path.join(save_folder, img_filename))
                sequence_input, img_input, = X
                self._forward_beam_search(X, beam_search_captions_encoded)
                img_encode_relevance= self._explain_sentence()
                x = int(np.sqrt(len(img_encode_relevance)))
                y = int(np.ceil(len(img_encode_relevance) / x))
                _, axes = plt.subplots(y, x, sharex="col", sharey="row", figsize=(20,20))
                axes = axes.flatten()
                for i in range(len(img_encode_relevance)):
                    relevance = self._explain_CNN(img_input, img_encode_relevance[i])
                    # print(beam_search_captions[0].split()[i])
                    # print(np.sum(relevance[0] == 0))
                    # print(np.sum(relevance[0] > 0))
                    # print(np.sum(relevance[0] < 0))
                    channels_first = K.image_data_format() == "channels_first"
                    color_conversion = None
                    hp = postprocess(relevance, color_conversion, channels_first)
                    hp = heatmap(hp)[0]
                    axes[i].set_title(self._preprocessor._word_of[beam_search_captions_encoded[i]], fontsize=18)
                    axes[i].imshow(hp)
                plt.show()
                plt.savefig(os.path.join(save_folder, img_filename.split('.')[0] + 'gradienttimesinput_hm.jpg'))
                break
class ExplainImgCaptioningGridTDGuidedGradcam(ExplainImgCaptioningGridTDGradient):
    def __init__(self,  model, weight_path, dataset_provider, max_caption_length=20):
        super(ExplainImgCaptioningGridTDGradient, self).__init__(model, weight_path, dataset_provider, max_caption_length)
        self._CNN_explainer= GuidedBackprop(self._image_model, neuron_selection_mode="replace")

    def _explain_CNN(self, X, relevance_value, ):
        gradcamp = self.grad_cam(self._image_features_input_bm, relevance_value[0])
        guided_backprop_relevance = self._CNN_explainer.analyze([X, relevance_value])
        # if you use guided gradcam use the guided_backprop line
        relevance = guided_backprop_relevance[0] * gradcamp[..., np.newaxis]
        # if explain gradcam, use the np.ones line
        # relevance = np.ones(guided_backprop_relevance.shape)[0] * gradcamp[..., np.newaxis]
        return relevance[np.newaxis,:]

    def grad_cam(self, img_feature, grads):
        weights = np.mean(grads, axis=(0, 1))
        conv_output = img_feature.reshape((int(np.sqrt(self.L)), int(np.sqrt(self.L)), self.D))
        cam = np.zeros(conv_output.shape[0:2], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * conv_output[:, :, i]
        cam = skimage.transform.pyramid_expand(cam, upscale=16, sigma=20,
                                               multichannel=False)
        cam = np.maximum(cam, 0)
        cam_heatmap = cam/(np.max(np.abs(cam)) + 1e-6)
        return cam_heatmap

    def guided_backproe(self, img_input, grads):
        guided_backpropagate_relevance = self._CNN_explainer.analyze([img_input, grads])
        return guided_backpropagate_relevance

    def process_beam_search(self, beam_size=3, ):
        print('Guided_gradcam')
        data_generator = self._dataset_provider.test_set(include_datum=True)
        count = 0
        for X, y, data in data_generator:
            imgs_path = map(attrgetter('img_path'), data)
            imgs_path = list(imgs_path)
            count += 1
            if count ==1:  # 9
                beam_search_captions_encoded = self._beam_search(X, beam_size)[0]
                beam_search_captions = self._preprocessor.decode_captions_from_list1d(beam_search_captions_encoded)
                img_filename = imgs_path[0].split('/')[-1]
                save_folder = os.path.join(self._weight_path + 'explanation', img_filename.split('.')[0])
                if not os.path.isdir(save_folder):
                    os.makedirs(save_folder)
                img_original = Image.open(imgs_path[0])
                img_original = img_original.resize((224, 224))
                img_original.save(os.path.join(save_folder, img_filename))
                sequence_input, img_input, = X
                self._forward_beam_search(X, beam_search_captions_encoded)
                '''for a full sentence'''
                img_encode_relevance= self._explain_sentence()
                x = int(np.sqrt(len(img_encode_relevance)))
                y = int(np.ceil(len(img_encode_relevance) / x))
                _, axes = plt.subplots(y, x, sharex="col", sharey="row", figsize=(20,20))
                # plt.axis('off')
                axes = axes.flatten()
                for i in range(len(img_encode_relevance)):
                    relevance = self._explain_CNN(img_input, img_encode_relevance[i])
                    # print(beam_search_captions[0].split()[i])
                    # print(np.sum(relevance[0] == 0))
                    # print(np.sum(relevance[0] > 0))
                    # print(np.sum(relevance[0] < 0))
                    # relevance = np.maximum(relevance, 0)
                    channels_first = K.image_data_format() == "channels_first"
                    color_conversion = "BGRtoRGB"
                    hp = postprocess(relevance, color_conversion, channels_first)
                    hp = heatmap(hp)[0]
                    axes[i].set_title(self._preprocessor._word_of[beam_search_captions_encoded[i]], fontsize=18)
                    axes[i].imshow(hp)
                plt.savefig(os.path.join(save_folder, img_filename.split('.')[0] + 'guidedgradcam_hm.jpg'))
                break

def main_gridTD_attention(config, dataset, model_weight_path, max_caption_length=20):
    dataset_provider = DatasetPreprocessorAttention(dataset, config, single_caption=True)
    model = ImgCaptioninggridTDAdaptiveModel(config)
    model.build(dataset_provider.vocabs, dataset_provider.vocab_size)
    explainer = ExplainImgCaptioningGridTDModel(model, weight_path=model_weight_path, dataset_provider=dataset_provider,
                                                max_caption_length=max_caption_length)
    explainer.process_beam_search(3, )
def main_gridTD_guidedgradcam(config, dataset, model_weight_path, max_caption_length=20):
    dataset_provider = DatasetPreprocessorAttention(dataset, config, single_caption=True)
    model = ImgCaptioninggridTDAdaptiveModel(config)
    model.build(dataset_provider.vocabs, dataset_provider.vocab_size)

    explainer = ExplainImgCaptioningGridTDGuidedGradcam(model, weight_path=model_weight_path, dataset_provider=dataset_provider,
                                                        max_caption_length=max_caption_length)
    explainer.process_beam_search(3)

def main_adaptive_attention(config, dataset, model_weight_path, max_caption_length=20):

    dataset_provider = DatasetPreprocessorAttention(dataset, config,single_caption=True)
    model = ImgCaptioningAdaptiveAttentionModel(config)
    model.build(dataset_provider.vocabs, dataset_provider.vocab_size)

    explainer = ExplainImgCaptioningAdaptiveAttention(model, weight_path=model_weight_path, dataset_provider=dataset_provider,
                                                      max_caption_length=max_caption_length)
    explainer.process_beam_search(3, )
def main_adaptive_attention_guidedgradcam(config, dataset, model_weight_path, max_caption_length=20):
    dataset_provider = DatasetPreprocessorAttention(dataset, config, single_caption=True)
    model = ImgCaptioningAdaptiveAttentionModel(config)
    model.build(dataset_provider.vocabs, dataset_provider.vocab_size)

    explainer = ExplainImgCaptioningAdaptiveAttentionGuidedGradcam(model, weight_path=model_weight_path, dataset_provider=dataset_provider,
                                                                   max_caption_length=max_caption_length)
    explainer.process_beam_search(3)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 0 :3 2:1 1:0 3:2
    flickr_config = config.FlickrConfig()
    flickr_config.batch_size=1
    dataset = Flickr30kDataset(flickr_config,single_caption=True)
    training_dir = '../results/flickr30k/training-results/flickr_VGG16_gridTD_attention/'
    # training_dir = './results/flickr30k/training-results/flickr_VGG16_adaptive_attention/'
    model_weight_path = os.path.join(training_dir, 'keras_model.hdf5')
    # main_gridTD_guidedgradcam(flickr_config, dataset, model_weight_path, max_caption_length=20)
    main_gridTD_attention(flickr_config, dataset, model_weight_path)
    # main_adaptive_attention_guidedgradcam(flickr_config, dataset, model_weight_path, max_caption_length=20)
    # main_adaptive_attention(flickr_config, dataset,model_weight_path)

