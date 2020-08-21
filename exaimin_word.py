from explainers import *
from model import *
import json
import random
import pickle
from io_utils import logging, write_yaml_file
from metrics import BLEU, CIDEr, METEOR, ROUGE, SPICE, BERT
import yaml
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

FLICKR_CATEGORY = ['people','man' , 'boy', 'girl', 'dog', 'jacket', 'shirt', 'hat', 'dress', 'ball', 'bicycle', 'microphone']

COCO_CATEGORY = ['bicycle','car','motorcycle','airplane','bus','train','truck','boat',
                  'bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','hat',
                  'umbrella','handbag','tie','suitcase','frisbee', 'skis', 'snowboard', 'kite',
                  'skateboard','surfboard','bottle','cup','fork','knife','spoon','bowl','banana','apple',
                  'sandwich','orange','broccoli','carrot','pizza','donut','cake','chair','couch','bed',
                  'toilet','tv','laptop','mouse','remote','keyboard','microwave', 'oven', 'toaster','sink','refrigerator',
                  'book','clock','vase','scissors','toothbrush',
                  'ball', 'bat', 'glove', 'racket', 'hydrant',  'glass', 'drier',  'table', 'phone']

FREQUENT_OBJECT = ['man','shirt', 'woman', 'people', 'group', 'street', 'dog', 'bench', 'boy']

COCO_FREQUENT_OBJECT=['man', 'group', 'people','street', 'table', 'woman', 'plate', 'tennis', 'food', 'train', 'person','road','sink', 'building','cat','bathroom'
                      ,'snow', 'baseball', 'bench','clock', 'dog','toilet', 'laptop','bus','computer', 'beach', 'court', 'skateboard','surfboard','desk', 'bed']


class Explainer(object):

    def __init__(self, model, weight_path, explainer, max_caption_length, beam_size):
        self._keras_model = model.keras_model
        self._keras_model.load_weights(weight_path)

        self._image_model = Model(inputs=self._keras_model.get_layer('input_1').input,
                                  outputs=self._keras_model.get_layer('block5_conv3').output)
        self._img_encoder = model.img_encoder
        self._CNN_explainer = LRPSequentialPresetA(self._image_model, epsilon=EPS,  neuron_selection_mode='replace')
        self._image_preprocessor = explainer._dataset_provider.image_preprocessor
        self._caption_preprocessor = explainer._dataset_provider.caption_preprocessor
        self._explainer = explainer
        self._max_caption_length = max_caption_length
        self._beam_size = beam_size
        if self._img_encoder in ['vgg16', 'vgg19']:
            self._color_conversion = 'BGRtoRGB'
            self._reshape_size = (14,14)
            self._upscale = 16
        elif self._img_encoder == 'inception_v3':
            self._color_conversion = None
            self._reshape_size = (5,5)
            self._upscale = 20
        else:
            raise NotImplementedError('the img_encode is not valid, [vgg16, vgg19, inception_v3]')

    def _preprocess_img(self, img_path):
        preprocessed_img = self._image_preprocessor.preprocess_images(img_path)
        img_array = self._image_preprocessor.preprocess_batch(preprocessed_img)
        initial_caption = self._caption_preprocessor.SOS_TOKEN_LABEL_ENCODED
        return (initial_caption, img_array)

    def _predict_caption(self, X):
        captions = self._explainer._beam_search(X, beam_size=self._beam_size)[0]

        return captions

    def _max_pooling(self, hp):
        '''do a max pooling with kernal size 6*6 to generate 14*14 output'''
        output = np.zeros((14,14))
        for i in range(0, 224,16):
            for j in range(0, 224, 16):
                output[int(i/16),int(j/16)] = np.max(hp[i:i+16, j:j+16])
        return output

    def _ave_pooling(self,hp):
        output = np.zeros((14,14))
        for i in range(0, 224, 16):
            for j in range(0, 224, 16):
                output[int(i/16), int(j/16)] = np.mean(hp[i:i+16, j:j+16])
        return output

    def _explain_single_word(self, X, captions, t):
        def project(x):
            absmax = np.max(np.abs(x))
            if absmax == 0:
                return np.zeros(x.shape)
            x = 1.0 * x/absmax
            # if np.sum(x<0):
            #     x = (x + 1)/2
            # else:
            #     x = x
            return x
        self._explainer._forward_beam_search(X, captions)
        beam_search_captions = self._caption_preprocessor.decode_captions_from_list1d(captions)
        sequence_input, img_input = X
        img_encode_relevance, attention = self._explainer._explain_lstm_single_word_sequence(t,rule='eps')
        relevance = self._explainer._explain_CNN(img_input, img_encode_relevance)
        channels_first = K.image_data_format() == "channels_first"
        hp= postprocess(relevance, self._color_conversion, channels_first)
        # print(hp.shape)


        hp = np.mean(hp, axis=-1)[0]
        # print(hp.shape)
        hp = project(hp)

        num_pos = np.sum(np.mean(hp, axis=-1) > 0)
        atn = skimage.transform.pyramid_expand(attention.reshape(self._reshape_size), upscale=self._upscale, sigma=20,  multichannel=False)
        # atn = atn[:, :, np.newaxis]
        atn = project(atn)
        # quantile = np.quantile(hp, [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
        # quantile = quantile.tolist()
        # print(hp.shape, atn.shape)
        return hp, atn

    def _get_explanation_single_word(self, X, captions, t):
        def project(x):
            absmax = np.max(np.abs(x))
            if absmax == 0:
                return np.zeros(x.shape)
            x = 1.0 * x/absmax
            return x
        self._explainer._forward_beam_search(X, captions)
        sequence_input, img_input = X
        img_encode_relevance, attention = self._explainer._explain_lstm_single_word_sequence(t, rule='eps')
        relevance = self._explainer._explain_CNN(img_input, img_encode_relevance)
        channels_first = K.image_data_format() == "channels_first"
        hp= postprocess(relevance, self._color_conversion, channels_first)

        hp = np.mean(hp, axis=-1)[0]
        hp = project(hp)
        return hp

    def _explain_single_word_pooling(self, X, captions, t, poolingtype='max'):
        def project(x):
            absmax = np.max(np.abs(x))
            if absmax == 0:
                return np.zeros(x.shape)
            x = 1.0 * x/absmax
            # if np.sum(x<0):
            #     x = (x + 1)/2
            # else:
            #     x = x
            return x
        self._explainer._forward_beam_search(X, captions)
        beam_search_captions = self._caption_preprocessor.decode_captions_from_list1d(captions)
        sequence_input, img_input = X
        img_encode_relevance, attention = self._explainer._explain_lstm_single_word_sequence(t,rule='eps')
        relevance = self._explainer._explain_CNN(img_input, img_encode_relevance)
        channels_first = K.image_data_format() == "channels_first"
        hp= postprocess(relevance, self._color_conversion, channels_first)
        hp = np.mean(hp, axis=-1)[0]
        if poolingtype == 'max':
            hp = self._max_pooling(hp)
        if poolingtype == 'ave':
            hp = self._ave_pooling(hp)
        hp = project(hp)
        # atn = atn[:, :, np.newaxis]
        atn = project(attention)
        # quantile = np.quantile(hp, [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
        # quantile = quantile.tolist()
        print(hp.shape, atn.shape)
        return hp, atn

    def analyze_single_word(self, img_path, t):
        self.img_path = img_path
        X = self._preprocess_img([img_path])
        captions = self._predict_caption(X)
        img_name = img_path.split('/')[-1]
        # print(img_name)
        hp, atn = self._explain_single_word(X, captions, t)
        return hp, atn

    def analyze_single_word_pooling(self, img_path, t, poolingtype='max'):
        self.img_path = img_path
        X = self._preprocess_img([img_path])
        captions = self._predict_caption(X)
        img_name = img_path.split('/')[-1]
        print(img_name)
        hp, atn = self._explain_single_word_pooling(X, captions, t, poolingtype=poolingtype)
        return hp, atn


class ExplainerGuidedgradcam(Explainer):
    def __init__(self, model, weight_path, explainer, max_caption_length, beam_size):
        super(ExplainerGuidedgradcam, self).__init__(model, weight_path, explainer, max_caption_length, beam_size)

    def _explain_captions(self, X, captions, save_folder):
        self._explainer._forward_beam_search(X, captions)
        beam_search_captions = self._caption_preprocessor.decode_captions_from_list1d(captions)
        print(beam_search_captions)
        img_encode_relevance, attention = self._explainer._explain_sentence(rule='eps')
        sequence_input, img_input = X
        x = int(np.sqrt(len(attention)))
        y = int(np.ceil(len(attention) / x))
        _, axes = plt.subplots(y, x, sharex="col", sharey="row", figsize=(20, 20))
        axes = axes.flatten()
        for i in range(len(attention)):
            relevance = self._explainer._explain_CNN(img_input, img_encode_relevance[i])
            channels_first = K.image_data_format() == "channels_first"
            hp = postprocess(relevance, self._color_conversion, channels_first)
            # hp = postprocess(relevance, color_conversion="BGRtoRGB", channels_first=False)

            hp = heatmap(hp)
            axes[i].set_title(self._caption_preprocessor._word_of[captions[i]], fontsize=18)
            axes[i].imshow(hp[0])
        plt.savefig(os.path.join(save_folder, 'guidedgradcam_hm.jpg'))
        x = int(np.sqrt(len(attention)))
        y = int(np.ceil(len(attention) / x))
        _, axes = plt.subplots(y, x, sharex="col", sharey="row", figsize=(20, 20))
        plt.axis('off')
        axes = axes.flatten()
        img_original = Image.open(self.img_path)
        blank = np.zeros((224, 224, 2))
        img_original = img_original.resize((224, 224))

        img_original.save(os.path.join(save_folder, save_folder.split('/')[-1]))
        # for i in range(len(attention)):
        #     if self._img_encoder == "inception_v3":
        #         atn = skimage.transform.pyramid_expand(attention[i].reshape(5,5), upscale=20, sigma=20,
        #                                                multichannel=False)
        #         atn = skimage.transform.resize(atn, (224, 224), mode='reflect', anti_aliasing=True)
        #     else:
        #         atn = skimage.transform.pyramid_expand(attention[i].reshape(14, 14), upscale=16, sigma=20,
        #                                                multichannel=False)
        #
        #     atn = atn[:, :, np.newaxis]
        #     atn = (atn - np.min(atn)) / np.max(atn) * 255
        #     # blank = np.zeros((224, 224, 2))
        #     atn = np.concatenate((atn, blank), axis=-1)
        #     attention_img = Image.fromarray(np.uint8(atn), img_original.mode)
        #     # attention_img.show()
        #     tmp_img = Image.blend(img_original, attention_img, 0.7)
        #     axes[i].set_title(self._caption_preprocessor._word_of[captions[i]], fontsize=18)
        #     axes[i].imshow(tmp_img)
        # plt.savefig(os.path.join(save_folder, 'attention.jpg'))

    def _explain_single_word(self, X, captions, t):
        def project(x):
            absmax = np.max(np.abs(x))
            if absmax == 0:
                return np.zeros(x.shape)
            x = 1.0 * x/absmax
            # if np.sum(x<0):
            #     x = (x + 1)/2
            # else:
            #     x = x
            return x
        self._explainer._forward_beam_search(X, captions)
        beam_search_captions = self._caption_preprocessor.decode_captions_from_list1d(captions)
        sequence_input, img_input = X
        img_encode_relevance, attention = self._explainer._explain_lstm_single_word_sequence(t,rule='eps')
        relevance = self._explainer._explain_CNN(img_input, img_encode_relevance)
        channels_first = K.image_data_format() == "channels_first"
        hp= postprocess(relevance, self._color_conversion, channels_first)
        hp = np.mean(hp, axis=-1)[0]
        hp = project(hp)
        return hp

    def _explain_single_word_pooling(self, X, captions, t, poolingtype='max'):
        def project(x):
            absmax = np.max(np.abs(x))
            if absmax == 0:
                return np.zeros(x.shape)
            x = 1.0 * x/absmax
            # if np.sum(x<0):
            #     x = (x + 1)/2
            # else:
            #     x = x
            return x
        self._explainer._forward_beam_search(X, captions)
        beam_search_captions = self._caption_preprocessor.decode_captions_from_list1d(captions)
        sequence_input, img_input = X
        img_encode_relevance, attention = self._explainer._explain_lstm_single_word_sequence(t,rule='eps')
        relevance = self._explainer._explain_CNN(img_input, img_encode_relevance)
        channels_first = K.image_data_format() == "channels_first"
        hp= postprocess(relevance, self._color_conversion, channels_first)
        hp = np.mean(hp, axis=-1)[0]
        if poolingtype == 'max':
            hp = self._max_pooling(hp)
        if poolingtype == 'ave':
            hp = self._ave_pooling(hp)
        hp = project(hp)
        return hp

    def analyze_single_word(self, img_path, t):
        self.img_path = img_path
        X = self._preprocess_img([img_path])
        captions = self._predict_caption(X)
        img_name = img_path.split('/')[-1]
        print(img_name)
        hp= self._explain_single_word(X, captions, t)
        return hp

    def analyze_single_word_pooling(self, img_path, t, poolingtype='max'):
        self.img_path = img_path
        X = self._preprocess_img([img_path])
        captions = self._predict_caption(X)
        img_name = img_path.split('/')[-1]
        print(img_name)
        hp= self._explain_single_word_pooling(X, captions, t, poolingtype=poolingtype)
        return hp


def preprocess_predicted_captions_yaml(prediction_path, file_name):
    '''This function read the saved prediction files into dictionary
    prediction_path: a path to the folder of a yaml file which contains the predicted caption of an image
    file_name: the name of the prediction file '''
    with open(os.path.join(prediction_path, file_name)) as f:
        key_captions = yaml.load(f)
        f.close()
    predict_dict = dict()
    for key in key_captions.keys():
        predict_dict[key] = key_captions[key][0]['caption']
    return predict_dict #{image_name: caption}


def analyze_predicted_captions(prediction_path, file_name, dataset, category_list):
    '''

    :param prediction_path: a path to the folder of the prediction file
    :param file_name: the name of the prediction file
    :param dataset: the dataset object, either flickr30K or coco
    :param category_list: the frequent appeared object words of the dataset
    :return: the mAP of the object words within the category_list
    '''
    count_cat = {}
    tp = {}
    for cat in category_list:
        count_cat[cat] = 0
        tp[cat] = 0.
    predict_dict = preprocess_predicted_captions_yaml(prediction_path, file_name)
    test_set = dataset.test_set
    acc_true = 0.   # number of words appearing in both the prediction and the reference caption
    total_true = 0.  # the number of words in the reference caption
    predict_true = 0. # the number of words in the predicted caption
    predict_list = []
    correct_predict_list = []
    for category in category_list:
        for item in test_set:
            if item.img_filename not in predict_dict.keys():
                continue
            predict_caption = predict_dict[item.img_filename]
            predict_words = predict_caption.split()
            true_captions = item.all_captions_txt
            acc_flag = False
            total_flag = False
            if category in predict_words:
                predict_true += 1
                predict_list.append(item.img_filename)
            for true_caption in true_captions:
                true_caption_words = true_caption.split()
                if category in true_caption_words and category in predict_words:
                    acc_flag = True
                    correct_predict_list.append(item.img_filename)
                if category in true_caption_words:
                    total_flag = True
            if acc_flag:
                acc_true += 1
                tp[category] += 1
            if total_flag:
                count_cat[category] += 1
                total_true += 1
    mAP = 0.
    print(count_cat)
    for key in count_cat.keys():
        tp[key] = tp[key]/count_cat[key]
        mAP += tp[key]
    print('the mAP of the frequent object words is:', mAP/(len(tp)-1) * 100)
    return predict_true, total_true, acc_true

'''get the lrp statistics of the top-9 frequent object words, prediction_path and file_name point to the file (image_id: caption)
category_list is the top-k frequent words'''
def analyze_beta_of_category_generate_adaptive(prediction_path, file_name, category_list, max_caption_length=20):
    def get_index(caption, category):
        words = caption.split(' ')
        for t in range(len(words)):
            if category == words[t]:
                return t+1
        return None

    model_weight_path = './results/flickr30k/training-results/flickr_VGG16_adaptive_attention/keras_model.hdf5'
    dataset = Flickr30kDataset(single_caption=True)
    flickr30_config = config.FlickrConfig()
    flickr30_config.batch_size = 1
    dataset_provider = DatasetPreprocessorAttention(dataset, flickr30_config, single_caption=True)
    model = ImgCaptioningAdaptiveAttentionModel(flickr30_config)
    model.build(dataset_provider.vocabs, dataset_provider.vocab_size)
    explainer_engine = ExplainImgCaptioningAdaptiveAttention(model, model_weight_path, dataset_provider,
                                                            max_caption_length)
    predict_dict = preprocess_predicted_captions_yaml(prediction_path, file_name)
    test_set = dataset.test_set
    save_dict = dict()
    max_num = len(test_set)
    count = 0
    for X, y, data in dataset_provider.test_set(include_datum=True):
        if count >= max_num:
            break
        predict_caption = predict_dict[data[0].img_filename]
        for category in category_list:
            if category in predict_caption:
                index = get_index(predict_caption,category)
                if index:
                    if data[0].img_filename not in save_dict.keys():
                        save_dict[data[0].img_filename] = dict()
                        save_dict[data[0].img_filename]['beta'] = []
                    save_dict[data[0].img_filename]['predict_caption'] = predict_caption
                    true_captions = data[0].all_captions_txt
                    save_dict[data[0].img_filename]['true_captions'] = true_captions
                    beam_search_captions_encoded = explainer_engine._beam_search(X, 3)[0]
                    explainer_engine._forward_beam_search(X, beam_search_captions_encoded)
                    print(predict_caption,category, index, len(predict_caption.split()))
                    print(len(explainer_engine.beta))
                    save_dict[data[0].img_filename]['beta'].append((category, explainer_engine.beta[index]))
        count += 1
    with open('./results/flickr30k/evaluation-results/flickr_VGG16_adaptive_attention/flickr30K_beta_analyze_category_top9.pkl', 'wb') as f:
        pickle.dump(save_dict, f)
        f.close()
def analyze_mean_of_category_generate_adaptive(prediction_path, file_name, category_list, max_caption_length=20):
    def get_index(caption, category):
        words = caption.split(' ')
        for t in range(len(words)):
            if category == words[t]:
                return t+1
        return None
    model_weight_path = './results/flickr30k/training-results/flickr_VGG16_adaptive_attention/keras_model.hdf5'
    dataset = Flickr30kDataset(single_caption=True)
    flickr30_config = config.FlickrConfig()
    dataset_provider = DatasetPreprocessorAttention(dataset, flickr30_config, single_caption=True)
    model = ImgCaptioningAdaptiveAttentionModel(flickr30_config)
    model.build(dataset_provider.vocabs, dataset_provider.vocab_size)
    explainer_engine = ExplainImgCaptioningAdaptiveAttention(model, model_weight_path, dataset_provider,
                                                            max_caption_length)

    explainer = Explainer(model, model_weight_path, explainer_engine, max_caption_length, beam_size=3)
    predict_dict = preprocess_predicted_captions_yaml(prediction_path, file_name)
    test_set = dataset.test_set
    save_dict = dict()
    for item in test_set:
        predict_caption = predict_dict[item.img_filename]
        words = predict_caption.split()
        for category in category_list:
            if category in words:
                index = get_index(predict_caption,category)
                if index:
                    if item.img_filename not in save_dict.keys():
                        save_dict[item.img_filename] = dict()
                        save_dict[item.img_filename]['lrp_mean'] = []
                        save_dict[item.img_filename]['attention_mean'] = []
                    save_dict[item.img_filename]['predict_caption'] = predict_caption
                    true_captions = item.all_captions_txt
                    save_dict[item.img_filename]['true_captions'] = true_captions
                    hp, atn = explainer.analyze_single_word(item.img_path, index)
                    save_dict[item.img_filename]['lrp_mean'].append((category, np.mean(hp)))
                    save_dict[item.img_filename]['attention_mean'].append((category, np.mean(atn)))
    with open('./results/flickr30k/evaluation-results/flickr_VGG16_adaptive_attention/flickr30k_analyze_category_mean_LRP_att_top9.pkl', 'wb') as f:
        pickle.dump(save_dict, f)
        f.close()
def analyze_mean_of_category_generate_adaptive_ggradcam(prediction_path, file_name, category_list, max_caption_length=20):
    def get_index(caption, category):
        words = caption.split(' ')
        for t in range(len(words)):
            if category == words[t]:
                return t+1
        return None
    model_weight_path = './results/flickr30k/training-results/flickr_VGG16_adaptive_attention/keras_model.hdf5'
    dataset = Flickr30kDataset(single_caption=True)
    flickr30_config = config.FlickrConfig()
    dataset_provider = DatasetPreprocessorAttention(dataset, flickr30_config, single_caption=True)
    model = ImgCaptioningAdaptiveAttentionModel(flickr30_config)
    model.build(dataset_provider.vocabs, dataset_provider.vocab_size)
    explainer_engine = ExplainImgCaptioningAdaptiveAttentionGuidedGradcam(model, model_weight_path, dataset_provider,
                                                            max_caption_length)
    explainer = ExplainerGuidedgradcam(model, model_weight_path, explainer_engine, max_caption_length, beam_size=3)
    predict_dict = preprocess_predicted_captions_yaml(prediction_path, file_name)
    test_set = dataset.test_set
    save_dict = dict()
    for item in test_set:
        predict_caption = predict_dict[item.img_filename]
        for category in category_list:
            if category in predict_caption:
                index = get_index(predict_caption,category)
                if index:
                    if item.img_filename not in save_dict.keys():
                        save_dict[item.img_filename] = dict()
                        save_dict[item.img_filename]['guidedgradcam_mean'] = []
                    save_dict[item.img_filename]['predict_caption'] = predict_caption
                    true_captions = item.all_captions_txt
                    save_dict[item.img_filename]['true_captions'] = true_captions
                    hp = explainer.analyze_single_word(item.img_path, index)
                    hp = np.abs(hp)
                    save_dict[item.img_filename]['guidedgradcam_mean'].append((category, np.mean(hp)))
    with open('./results/flickr30k/evaluation-results/onelayer_VGG16_local_attentionv3/flickr30k_analyze_category_mean_guidedgradcam_abs_top9.pkl', 'wb') as f:
        pickle.dump(save_dict, f)
        f.close()
def analyze_beta_of_category_generate_gridTD(prediction_path, file_name, category_list, max_caption_length=20):
    def get_index(caption, category):
        words = caption.split(' ')
        for t in range(len(words)):
            if category == words[t]:
                return t+1
        return None

    model_weight_path = './results/flickr30k/training-results/flickr_VGG16_gridTD_attention/keras_model.hdf5'
    flickr30_config = config.FlickrConfig()
    flickr30_config.batch_size = 1
    dataset = Flickr30kDataset(flickr30_config,single_caption=True)
    dataset_provider = DatasetPreprocessorAttention(dataset, flickr30_config, single_caption=True)
    model = ImgCaptioninggridTDAdaptiveModel(flickr30_config)
    model.build(dataset_provider.vocabs, dataset_provider.vocab_size)
    explainer_engine = ExplainImgCaptioningGridTDModel(model, model_weight_path, dataset_provider,
                                                       max_caption_length)
    predict_dict = preprocess_predicted_captions_yaml(prediction_path, file_name)
    test_set = dataset.test_set
    save_dict = dict()
    max_num = len(test_set)
    count = 0
    for X, y, data in dataset_provider.test_set(include_datum=True):
        if count >= max_num:
            break
        predict_caption = predict_dict[data[0].img_filename]
        for category in category_list:
            if category in predict_caption:
                index = get_index(predict_caption,category)
                if index:
                    if data[0].img_filename not in save_dict.keys():
                        save_dict[data[0].img_filename] = dict()
                        save_dict[data[0].img_filename]['beta'] = []
                    save_dict[data[0].img_filename]['predict_caption'] = predict_caption
                    true_captions = data[0].all_captions_txt
                    save_dict[data[0].img_filename]['true_captions'] = true_captions
                    beam_search_captions_encoded = explainer_engine._beam_search(X, 3)[0]
                    explainer_engine._forward_beam_search(X, beam_search_captions_encoded)
                    print(predict_caption,category, index, len(predict_caption.split()))
                    print(len(explainer_engine.beta))
                    save_dict[data[0].img_filename]['beta'].append((category, explainer_engine.beta[index]))
        count += 1
    with open('./results/flickr30k/evaluation-results/flickr_VGG16_gridTD_attention/flickr30K_beta_analyze_category_top9.pkl', 'wb') as f:
        pickle.dump(save_dict, f)
        f.close()
def analyze_mean_of_category_generate_gridTD(prediction_path, file_name, category_list, max_caption_length=20):
    def get_index(caption, category):
        words = caption.split(' ')
        for t in range(len(words)):
            if category == words[t]:
                return t+1
        return None
    model_weight_path = './results/flickr30k/training-results/flickr_VGG16_gridTD_attention/keras_model.hdf5'
    flickr30_config = config.FlickrConfig()
    flickr30_config.batch_size = 1
    dataset = Flickr30kDataset(flickr30_config,single_caption=True)
    dataset_provider = DatasetPreprocessorAttention(dataset, flickr30_config, single_caption=True)
    model = ImgCaptioninggridTDAdaptiveModel(flickr30_config)
    model.build(dataset_provider.vocabs, dataset_provider.vocab_size)
    explainer_engine = ExplainImgCaptioningGridTDModel(model, model_weight_path, dataset_provider,
                                                       max_caption_length)

    explainer = Explainer(model, model_weight_path, explainer_engine, max_caption_length, beam_size=3)
    predict_dict = preprocess_predicted_captions_yaml(prediction_path, file_name)
    test_set = dataset.test_set
    save_dict = dict()
    for item in test_set:
        predict_caption = predict_dict[item.img_filename]
        words = predict_caption.split()
        for category in category_list:
            if category in words:
                index = get_index(predict_caption,category)
                if index:
                    if item.img_filename not in save_dict.keys():
                        save_dict[item.img_filename] = dict()
                        save_dict[item.img_filename]['lrp_mean'] = []
                        save_dict[item.img_filename]['attention_mean'] = []
                    save_dict[item.img_filename]['predict_caption'] = predict_caption
                    true_captions = item.all_captions_txt
                    save_dict[item.img_filename]['true_captions'] = true_captions
                    hp, atn = explainer.analyze_single_word(item.img_path, index)
                    save_dict[item.img_filename]['lrp_mean'].append((category, np.mean(hp)))
                    save_dict[item.img_filename]['attention_mean'].append((category, np.mean(atn)))
    with open('./results/flickr30k/evaluation-results/flickr_VGG16_gridTD_attention/flickr30k_analyze_category_mean_LRP_att_top9.pkl', 'wb') as f:
        pickle.dump(save_dict, f)
        f.close()
def analyze_mean_of_category_generate_gridTD_ggradcam(prediction_path, file_name, category_list, max_caption_length=20):
    def get_index(caption, category):
        words = caption.split(' ')
        for t in range(len(words)):
            if category == words[t]:
                return t+1
        return None
    model_weight_path = './results/flickr30k/training-results/flickr_VGG16_gridTD_attention/keras_model.hdf5'
    flickr30_config = config.FlickrConfig()
    flickr30_config.batch_size = 1
    dataset = Flickr30kDataset(flickr30_config,single_caption=True)
    dataset_provider = DatasetPreprocessorAttention(dataset, flickr30_config, single_caption=True)
    model = ImgCaptioninggridTDAdaptiveModel(flickr30_config)
    model.build(dataset_provider.vocabs, dataset_provider.vocab_size)
    explainer_engine = ExplainImgCaptioningGridTDGuidedGradcam(model, model_weight_path, dataset_provider,
                                                               max_caption_length)

    explainer = ExplainerGuidedgradcam(model, model_weight_path, explainer_engine, max_caption_length, beam_size=3)
    predict_dict = preprocess_predicted_captions_yaml(prediction_path, file_name)
    test_set = dataset.test_set
    save_dict = dict()
    for item in test_set:
        predict_caption = predict_dict[item.img_filename]
        for category in category_list:
            if category in predict_caption:
                index = get_index(predict_caption,category)
                if index:
                    if item.img_filename not in save_dict.keys():
                        save_dict[item.img_filename] = dict()
                        save_dict[item.img_filename]['guidedgradcam_mean'] = []
                    save_dict[item.img_filename]['predict_caption'] = predict_caption
                    true_captions = item.all_captions_txt
                    save_dict[item.img_filename]['true_captions'] = true_captions
                    hp = explainer.analyze_single_word(item.img_path, index)
                    hp = np.abs(hp)
                    save_dict[item.img_filename]['guidedgradcam_mean'].append((category, np.mean(hp)))
    with open('./results/flickr30k/evaluation-results/onelayer_VGG16_adaptive_bottomup/flickr30k_analyze_category_mean_guidedgradcam_abs_top9.pkl', 'wb') as f:
        pickle.dump(save_dict, f)
        f.close()




'''calculate the roc curve and auc score with mean beta and quantile, input  file_path and file_name are the statistics dictionary of the explanations'''
def analyze_category_beta_roc_auc(file_path,file_name):
    with open(file_path + file_name, 'rb') as f:
        dicts = pickle.load(f)
        f.close()
    label = []
    TP_count = 0
    beta_score = []
    FP_count = 0
    for key in dicts.keys():
        true_captions = dicts[key]['true_captions']
        for item in dicts[key]['beta']:
            category = item[0]
            TP_flag = False
            for cap in true_captions:
                words = cap.split()
                if category in words:
                    TP_flag = True
            if TP_flag:
                TP_count += 1
                label.append(1)
                beta_score.append(1-item[1])
            else:
                FP_count += 1
                label.append(0)
                beta_score.append(1-item[1])

    beta_fpr, beta_tpr, beta_threshold = roc_curve(label, beta_score)
    beta_roc_auc = auc(beta_fpr, beta_tpr)
    print(beta_roc_auc)
    print(TP_count, FP_count)
    return beta_fpr, beta_tpr
def analyze_category_mean_roc_auc(file_path, file_name):
    with open(file_path + file_name, 'rb') as f:
        dicts = pickle.load(f)
        f.close()
    lrp_label = []
    lrp_score = []
    attention_label = []
    attention_score = []
    TP_count = 0
    FP_count = 0
    for key in dicts.keys():
        true_captions = dicts[key]['true_captions']
        for item in dicts[key]['lrp_mean']:
            category = item[0]
            TP_flag = False
            for cap in true_captions:
                words = cap.split()
                if category in words:
                    TP_flag = True
            if TP_flag:
                TP_count += 1
                lrp_label.append(1)
                lrp_score.append(item[1])
            else:
                FP_count += 1
                lrp_label.append(0)
                lrp_score.append(item[1])
        for item in dicts[key]['attention_mean']:
            category = item[0]
            TP_flag = False
            for cap in true_captions:
                words = cap.split()
                if category in words:
                    TP_flag = True
            if TP_flag:
                attention_label.append(1)
                attention_score.append(item[1])
            else:
                attention_label.append(0)
                attention_score.append(item[1])
    lrp_fpr, lrp_tpr, lrp_threshold = roc_curve(lrp_label, lrp_score)
    lrp_roc_auc = auc(lrp_fpr, lrp_tpr)
    attention_fpr, attention_tpr, attention_threshold = roc_curve(attention_label, attention_score)
    attention_roc_auc = auc(attention_fpr, attention_tpr)
    save_data_lrp = np.zeros((2,len(lrp_fpr)))
    save_data_lrp[0] = lrp_fpr
    save_data_lrp[1] = lrp_tpr
    save_data_att = np.zeros((2,len(attention_fpr)))
    save_data_att[0] = attention_fpr
    save_data_att[1] = attention_tpr
    print(lrp_roc_auc, attention_roc_auc)
    print(TP_count, FP_count)
    return lrp_fpr, lrp_tpr, attention_fpr, attention_tpr
def analyze_category_mean_roc_auc_gradcam(file_path, file_name):
    with open(file_path + file_name, 'rb') as f:
        dicts = pickle.load(f)
        f.close()

    label = []
    grad_score = []
    TP_count = 0
    FP_count = 0
    for key in dicts.keys():
        true_captions = dicts[key]['true_captions']
        for item in dicts[key]['guidedgradcam_mean']:
            category = item[0]
            TP_flag = False
            for cap in true_captions:
                words = cap.split()
                if category in words:
                    TP_flag = True
            if TP_flag:
                TP_count += 1
                label.append(1)
                grad_score.append(item[1])
            else:
                FP_count += 1
                label.append(0)
                grad_score.append(item[1])

    grad_fpr, grad_tpr, grad_threshold = roc_curve(label, grad_score)
    grad_roc_auc = auc(grad_fpr, grad_tpr)
    save_data_grad = np.zeros((2,len(grad_fpr)))
    save_data_grad[0] = grad_fpr
    save_data_grad[1] = grad_tpr
    print(TP_count, FP_count)
    print(grad_roc_auc)
    # np.savetxt(os.path.join(file_path, 'guidedgradcam_mean_roc_top9.csv'), save_data_grad, delimiter=',')
    return grad_fpr, grad_tpr



'''given the predictions (image_id: caption), output the evaluation metrics of the captioning task.'''
def calculate_metrics(id_to_prediction_path, id_to_references_path, model_name):
    # id_to_prediction = pickle.load(open(id_to_prediction_path, 'rb'))
    id_to_prediction = yaml.safe_load(open(id_to_prediction_path,'r'))
    id_to_references = pickle.load(open(id_to_references_path, 'rb'))
    metrics_list = [BLEU(4), METEOR(), CIDEr(), ROUGE(), SPICE(), BERT()]
    metrics_value = {}
    for metric in metrics_list:
        metric_name_to_value = metric.calculate(id_to_prediction,
                                                id_to_references)
        metrics_value.update(metric_name_to_value)

    save_path = './results/Flickr30KfusedFashion1101/evaluation-results/'
    name = os.path.join(save_path, 'test-fused-metrics-{}-{}-{}-cloth_unbalance.yaml'.format(3, 20, model_name))
    write_yaml_file(metrics_value, name)

'''Get the top-k frequent words of the predicted captions'''
def count_frequent_words(predict_path, file_name):
    predict_dict = preprocess_predicted_captions_yaml(predict_path, file_name)
    vocab = {}
    for key in predict_dict.keys():
        predict_caption = predict_dict[key]
        words = predict_caption.split()
        for word in words:
            if word not in vocab:
                vocab[word] = 0
            vocab[word] += 1
    sorted_vocab = sorted(vocab.items(), key=lambda a: a[1])
    print(sorted_vocab)
    return vocab






if __name__ == '__main__':

    '''============calculate the mAP of frequent object words in flickr30K'''
    # test_config = config.FlickrConfig()
    # test_config.batch_size = 2
    # test_dataset = Flickr30kDataset(test_config, single_caption=True)
    # predicted_path = './results/flickr30k/training-results/flickr_VGG16_adaptive_lrp_inference_0.5_0.5'  #53.06
    # predicted_path = './results/flickr30k/training-results/flickr_VGG16_adaptive_lrp_inference_baseline' #51.60
    # predicted_path = './results/flickr30k/training-results/flickr_VGG16_gridTD_lrp_inference_0.5_0.5'  #55.80
    # predicted_path = './results/flickr30k/training-results/flickr_VGG16_gridTD_lrp_inference_baseline' #55.79
    # predicted_file_name = 'test-predictions-3-keras_model.hdf5-flickr30k-20.yaml'
    # predict_true, total_true, acc_true = analyze_predicted_captions(predicted_path, predicted_file_name, test_dataset, FREQUENT_OBJECT)

    '''============calculate the mAP of frequent object words in flickr30K'''
    # test_config = config.COCOConfig()
    # test_config.batch_size = 2
    # test_dataset = COCODataset(single_caption=True)
    # predicted_path = './results/coco/training-results/coco_VGG16_adaptive_lrp_inference_0.5_0.5'  # 64.58
    # predicted_path = './results/coco/training-results/coco_VGG16_adaptive_lrp_inference_baseline' #63.53
    # predicted_path = './results/coco/training-results/coco_VGG16_gridTD_lrp_inference_0.5_0.5'  #64.19
    # predicted_path = './results/coco/training-results/coco_VGG16_gridTD_lrp_inference_baseline' #64.13
    # predicted_file_name = 'test-predictions-3-keras_model.hdf5-coco-20.yaml'
    # predict_true, total_true, acc_true = analyze_predicted_captions(predicted_path, predicted_file_name, test_dataset, COCO_FREQUENT_OBJECT)

    '''get the statistics of the image explanations(mean LRP, mean attention, abs GuidedGradcam etc)'''
    ## an example
    # result_dir = './results/flickr30k/training-results/flickr_VGG16_adaptive_attention/'
    # file_name = 'test-predictions-3-keras_model.hdf5-flickr30k-20.yaml'
    # analyze_beta_of_category_generate_adaptive(result_dir, file_name, FREQUENT_OBJECT)



    '''after generate the statistics files, we can plot the roc_auc curve of the statistics'''
    # file_path = './results/flickr30k/evaluation-results/flickr_VGG16_gridTD_attention/'
    # # file_path = './results/flickr30k/evaluation-results/flickr_VGG16_adaptive_attention/'
    # file_name = 'flickr30k_analyze_category_mean_LRP_att_top9.pkl'
    # lrp_x, lrp_y, attention_x, attention_y = analyze_category_mean_roc_auc(file_path, file_name)
    # file_name = 'flickr30k_analyze_category_mean_guidedgradcam_top9.pkl'
    # grad_x, grad_y = analyze_category_mean_roc_auc_gradcam(file_path, file_name)
    # file_name = 'flickr30K_beta_analyze_category_top9.pkl'
    # beta_x, beta_y = analyze_category_beta_roc_auc(file_path, file_name)
    # file_name = 'flickr30k_analyze_category_mean_LRP_pos_att_top9.pkl'
    # lrp_x_pos, lrp_y_pos, _, _ = analyze_category_mean_roc_auc(file_path, file_name)
    # file_name = 'flickr30k_analyze_category_mean_guidedgradcam_abs_top9.pkl'
    # grad_x_abs, grad_y_abs = analyze_category_mean_roc_auc_gradcam(file_path, file_name)

    # '''=============draw lines============'''
    # csfont = {'fontname': 'Times New Roman'}
    # fig = plt.figure(figsize=(10,7))
    # plt.rc('font', size=18)
    # ax = plt.axes()
    # plt.plot(lrp_x, lrp_y, color='red', label='LRP')
    # plt.plot(attention_x, attention_y, color='green', label='Attenton')
    # plt.plot(grad_x, grad_y, color='blue', label='Guided Grad-CAM')
    # plt.plot(beta_x, beta_y, color='yellow', label='1-beta')
    # plt.plot(lrp_x_pos, lrp_y_pos, color='orange', label='LRP-max')
    # plt.plot(grad_x_abs, grad_y_abs, color='Purple', label='Guided Grad-CAM-abs')
    # plt.xlabel('FPR',**csfont)
    # plt.ylabel('TPR', **csfont)
    # plt.legend(loc=2, fontsize=14)
    # plt.savefig(os.path.join(file_path, 'FPTR_roc_auc_top9_gridTD.png'), bbox_inches='tight')





