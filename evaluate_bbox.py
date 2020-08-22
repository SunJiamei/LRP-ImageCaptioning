import skimage.transform
import config
from models.explainers import *
from models.preparedataset import COCODataset, Flickr30kDataset, DatasetPreprocessorAttention
import json
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from models.model import *
import csv

CATEGORY_EXTENSION = {'person': ['people', 'woman', 'women', 'man', 'men', 'boy', 'girl', 'player', 'baby', 'person'],
                      'airplane':['plane', 'jetliner', 'jet', 'airplane'],
                      'bicycle':['bike', 'bicycle'],
                      'car':['car', 'taxi'],
                      }
PERSON = ['people', 'woman', 'women', 'man', 'men', 'boy', 'girl', 'player', 'baby', 'person']
AIRPLANE = ['plane', 'jetliner', 'jet', 'airplane']
BICYCLE = ['bike', 'bicycle']
CAR = ['car', 'taxi']
# words_categories = namedtuple('words2categories', 'word_list categories_id')
FILTER = ['a', 'A', 'an', 'An', 'the', 'The', '\'s']
Category_FILTER = ['a', 'A', 'an', 'An', 'the', 'The']

def show_bbox(img_path, bboxes, title, ratio,):
    img = Image.open(img_path)
    img = img.resize((224, 224), Image.ANTIALIAS)
    draw = ImageDraw.Draw(img)
    print(ratio)
    for bbox in bboxes:
        xmin = bbox[0] * ratio[0]
        ymin = bbox[1] * ratio[1]
        xmax = bbox[2] * ratio[0]
        ymax = bbox[3] * ratio[1]
        draw.rectangle([xmin, ymin, xmax, ymax], outline='black')
        draw.text((xmin, ymin), title)
    img.show()


class EvaluationBboxCOCO(object):
    def __init__(self, category_dict, max_caption_length, beam_size, rule, img_encode, explainer):
        self._preprocessor = explainer._preprocessor
        self._max_caption_length = max_caption_length
        self._beam_size = beam_size
        self._rule = rule
        self._category_dict = category_dict
        self._explainer = explainer
        self._img_encoder = img_encode
        if img_encode in ['vgg16', 'vgg19']:
            self._color_conversion = 'BGRtoRGB'
            self._reshape_size = (14,14)
            self._upscale = 20
        elif img_encode == 'inception_v3':
            self._color_conversion = None
            self._reshape_size = (5,5)
            self._upscale = 44
        else:
            raise NotImplementedError('the img_encode is not valid, [vgg16, vgg19, inception_v3]')

    def _get_explanation(self, X, t):
        def project(x):
            absmax = np.max(np.abs(x))
            if absmax == 0:
                return np.zeros(x.shape)
            x = 1.0 * x/absmax
            if np.sum(x<0):
                x = (x + 1)/2
            else:
                x = x
            return x

        img_encode_relevance, attention = self._explainer._explain_lstm_single_word_sequence(t)
        sequence_input, img_input, = X
        img_relevance = self._explainer._explain_CNN(img_input, img_encode_relevance)
        # print('img_relevance', img_relevance.shape)
        if self._img_encoder == 'inceptionv3':
            atn = skimage.transform.pyramid_expand(attention.reshape(5, 5), upscale=20, sigma=20, multichannel=False)
            atn = skimage.transform.resize(atn, (224, 224), mode='reflect', anti_aliasing=True)
        else:
            atn = skimage.transform.pyramid_expand(attention.reshape(14, 14), upscale=16, sigma=20, multichannel=False)
        hm = self._postprocess_images(img_relevance, color_coding=self._color_conversion, channels_first=False)
        hm = -1 * hm        # for negative score,   comment this line if using the positive scores to evaluate
        hm = np.maximum(hm, 0)
        hm = np.mean(hm, axis=-1)
        hm = project(hm)
        atn = project(atn)
        return hm[0], atn

    def _postprocess_images(self, images, color_coding=None, channels_first=None):

        ret = images
        image_data_format = K.image_data_format()
        assert color_coding in [None, "RGBtoBGR", "BGRtoRGB"]
        if color_coding in ["RGBtoBGR", "BGRtoRGB"]:
            if image_data_format == "channels_first":
                ret = ret[:, ::-1, :, :]
            if image_data_format == "channels_last":
                ret = ret[:, :, :, ::-1]

        if image_data_format == "channels_first" and not channels_first:
            ret = ret.transpose(0, 2, 3, 1)
        if image_data_format == "channels_last" and channels_first:
            ret = ret.transpose(0, 3, 1, 2)

        return ret

    def _project(self, X, output_range=(0, 1), absmax=None, input_is_postive_only=False):

        if absmax is None:
            absmax = np.max(np.abs(X),
                            axis=tuple(range(1, len(X.shape))))
        absmax = np.asarray(absmax)

        mask = absmax != 0
        if mask.sum() > 0:
            X[mask] /= absmax[mask]

        if input_is_postive_only is False:
            X = (X + 1) / 2  # [0, 1]
        X = X.clip(0, 1)

        X = output_range[0] + (X * (output_range[1] - output_range[0]))
        return X


    def _heatmap(self, X, cmap_type="seismic", reduce_op="sum", reduce_axis=-1, **kwargs):
        cmap = plt.cm.get_cmap(cmap_type)

        tmp = X
        shape = tmp.shape

        if reduce_op == "sum":
            tmp = tmp.sum(axis=reduce_axis)
        elif reduce_op == "absmax":
            pos_max = tmp.max(axis=reduce_axis)
            neg_max = (-tmp).max(axis=reduce_axis)
            abs_neg_max = -neg_max
            tmp = np.select([pos_max >= abs_neg_max, pos_max < abs_neg_max],
                            [pos_max, neg_max])
        else:
            raise NotImplementedError()

        tmp = self._project(tmp, output_range=(0, 255), **kwargs).astype(np.int64)

        tmp = cmap(tmp.flatten())[:, :3].T
        tmp = tmp.T

        shape = list(shape)
        shape[reduce_axis] = 3
        return tmp.reshape(shape).astype(np.float32)

    def _gamma(self, X, gamma=0.5, minamp=0, maxamp=None):
        """
        apply gamma correction to an input array X
        while maintaining the relative order of entries,
        also for negative vs positive values in X.
        the fxn firstly determines the max
        amplitude in both positive and negative
        direction and then applies gamma scaling
        to the positive and negative values of the
        array separately, according to the common amplitude.

        :param gamma: the gamma parameter for gamma scaling
        :param minamp: the smallest absolute value to consider.
        if not given assumed to be zero (neutral value for relevance,
            min value for saliency, ...). values above and below
            minamp are treated separately.
        :param maxamp: the largest absolute value to consider relative
        to the neutral value minamp
        if not given determined from the given data.
        """

        # prepare return array
        Y = np.zeros_like(X)

        X = X - minamp  # shift to given/assumed center
        if maxamp is None: maxamp = np.abs(X).max()  # infer maxamp if not given
        X = X / maxamp  # scale linearly

        # apply gamma correction for both positive and negative values.
        i_pos = X >= 0
        i_neg = np.invert(i_pos)
        Y[i_pos] = X[i_pos] ** gamma
        Y[i_neg] = -(-X[i_neg]) ** gamma

        # reconstruct original scale and center
        Y *= maxamp
        Y += minamp

        return Y

    def _calculate_overlaped_pixels(self, bbox, relevance, threshold):

        '''threshold is a scalar between [0,1]'''

        bbox_mask = np.zeros(relevance.shape)
        bbox_mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1
        relevance_mask = relevance <= threshold
        if np.sum(relevance_mask>0):
            relevance[relevance_mask] = 0
        total_pixel_score = np.sum(relevance)
        if total_pixel_score == 0:
            return 0
        correct_pixel_score = np.sum(np.multiply(bbox_mask, relevance))
        ratio = 1.0 * correct_pixel_score / total_pixel_score
        if ratio > 1:
            return 1.
        else:
            return ratio

    def _evaluate_per_img(self, X, category_dict):
        '''
           category_dict is a dictionary with keys as 'shpae','resize_ratio', 'categories', 'bbox'
           'categories' is a dictionary with key-value pairs as category_name-category_id
           'bbox' is a dictionary with key-value pairs as category_id - all boundinig boxes
           '''
        caption = self._explainer._beam_search(X, beam_size=self._beam_size)[0]
        self._explainer._forward_beam_search(X, caption)
        categories = category_dict['categories']
        bboxes = category_dict['bbox']
        resize_ratio = category_dict['resize_ratio']
        words_categories = dict()   # key is the id of category or phrase, value is a tuple (t, word)
        category_key = dict()
        for word_idx in range(len(caption)-1):
            word = self._preprocessor._word_of[caption[word_idx]]   # decode the caption encode into vocab
            for key in categories.keys():
                if key in CATEGORY_EXTENSION.keys():
                    if word not in FILTER and word in CATEGORY_EXTENSION[key]:
                        if categories[key] not in words_categories.keys():
                            words_categories[categories[key]] = set()
                            category_key[categories[key]] = key
                        words_categories[categories[key]].add((word_idx + 1, word))
                if word not in FILTER and word in key.split():
                    if categories[key] not in words_categories.keys():
                        words_categories[categories[key]] = set()
                        category_key[categories[key]] = key
                    words_categories[categories[key]].add((word_idx+1, word))  #  category_id --- (index in caption, word)
        lrp_relevance_pixel = dict()
        attention_pixel = dict()
        for key in words_categories.keys():
            bbox = bboxes[key]
            words = words_categories[key]
            lrp_relevance_pixel[key] = dict()
            attention_pixel[key] = dict()
            for item in words:
                t = item[0]
                relevance, attention = self._get_explanation(X, t)
                for box in bbox:
                    new_box = [0] * 4
                    new_box[0] = int(box[0] * resize_ratio[0])
                    new_box[1] = int(box[1] * resize_ratio[1])
                    new_box[2] = int(box[2] * resize_ratio[0])
                    new_box[3] = int(box[3] * resize_ratio[1])
                    for threshold in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                        if str(threshold) not in lrp_relevance_pixel[key].keys():
                            lrp_relevance_pixel[key][str(threshold)] = 0
                            attention_pixel[key][str(threshold)] = 0
                        lrp_score = self._calculate_overlaped_pixels(new_box,relevance,threshold)
                        attention_score = self._calculate_overlaped_pixels(new_box, attention, threshold)
                        if lrp_score > lrp_relevance_pixel[key][str(threshold)]:
                            lrp_relevance_pixel[key][str(threshold)] = lrp_score
                        if attention_score > attention_pixel[key][str(threshold)]:
                            attention_pixel[key][str(threshold)] = attention_score
        return lrp_relevance_pixel, attention_pixel, category_key

    def evaluate(self, X, data):

        img_filename = data[0].img_filename
        category = self._category_dict[img_filename]
        lrp_score, attention_score, category_key = self._evaluate_per_img(X, category)
        print('lrp_score', lrp_score)
        print('attention_score', attention_score)

        return lrp_score, attention_score, category_key
class EvaluationBboxCOCOBaseline(EvaluationBboxCOCO):

    def _get_explanation(self, X, t):
        print('baseline')
        def project(x):
            absmax = np.max(np.abs(x))
            if absmax == 0:
                return np.zeros(x.shape)
            x = 1.0 * x/absmax
            if np.sum(x<0):
                x = (x + 1)/2
            else:
                x = x
            return x
        img_encode_relevance = self._explainer._lstm_decoder_backward(t)
        sequence_input, img_input, = X
        img_relevance = self._explainer._explain_CNN(img_input, img_encode_relevance)
        hm = self._postprocess_images(img_relevance, color_coding=self._color_conversion, channels_first=False)
        #positive
        hm = -1 * hm         #negative
        hm = np.maximum(hm, 0)
        hm = np.mean(hm, axis=-1)
        hm = project(hm)
        return hm[0]

    def _evaluate_per_img(self, X, category_dict):
        '''
           category_dict is a dictionary with keys as 'shpae','resize_ratio', 'categories', 'bbox'
           'categories' is a dictionary with key-value pairs as category_name-category_id
           'bbox' is a dictionary with key-value pairs as category_id - all boundinig boxes
           '''
        caption = self._explainer._beam_search(X, beam_size=self._beam_size)[0]
        self._explainer._forward_beam_search(X, caption)
        categories = category_dict['categories']
        bboxes = category_dict['bbox']
        resize_ratio = category_dict['resize_ratio']
        words_categories = dict()   # key is the id of category or phrase, value is a tuple (t, word)
        category_key = dict()
        for word_idx in range(len(caption)-1):
            word = self._preprocessor._word_of[caption[word_idx]]   # decode the caption encode into vocab
            for key in categories.keys():
                if key in CATEGORY_EXTENSION.keys():
                    if word not in FILTER and word in CATEGORY_EXTENSION[key]:
                        if categories[key] not in words_categories.keys():
                            words_categories[categories[key]] = set()
                            category_key[categories[key]] = key
                        words_categories[categories[key]].add((word_idx + 1, word))
                if word not in FILTER and word in key.split():
                    if categories[key] not in words_categories.keys():
                        words_categories[categories[key]] = set()
                        category_key[categories[key]] = key
                    words_categories[categories[key]].add((word_idx+1, word))  #  category_id --- (index in caption, word)
        gradient_relevance_pixel = dict()
        for key in words_categories.keys():
            bbox = bboxes[key]
            words = words_categories[key]
            gradient_relevance_pixel[key] = dict()
            for item in words:
                t = item[0]
                relevance = self._get_explanation(X, t)
                # print(np.sum(relevance - attention))
                for box in bbox:
                    new_box = [0] * 4
                    new_box[0] = int(box[0] * resize_ratio[0])
                    new_box[1] = int(box[1] * resize_ratio[1])
                    new_box[2] = int(box[2] * resize_ratio[0])
                    new_box[3] = int(box[3] * resize_ratio[1])
                    for threshold in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                        if str(threshold) not in gradient_relevance_pixel[key].keys():
                            gradient_relevance_pixel[key][str(threshold)] = 0
                        gradient_score = self._calculate_overlaped_pixels(new_box,relevance,threshold)
                        if gradient_score > gradient_relevance_pixel[key][str(threshold)]:
                            gradient_relevance_pixel[key][str(threshold)] = gradient_score

        return gradient_relevance_pixel, category_key

    def evaluate(self, X, data):

        img_filename = data[0].img_filename
        category = self._category_dict[img_filename]
        gradient_score, category_key = self._evaluate_per_img(X, category)
        print('gradient_score', gradient_score)

        return gradient_score, category_key


def evaluate_coco_adaptive(max_caption_length=20):
    category_path = './dataset/coco/COCOvalEntities.json'
    model_weight_path= './results/coco/training-results/coco_training_onelayer_VGG16_local_attentionv3_finetune3/keras_model_05_0.7816.hdf5'
    dataset = COCODataset(single_caption=True)
    COCO_config = config.COCOConfig()
    COCO_config.batch_size = 1
    dataset_provider = DatasetPreprocessorAttention(dataset, COCO_config, single_caption=True)

    with open(category_path, 'r') as f:
        category_dict = json.load(f)
        f.close()
    cnn_type = COCO_config.img_encoder
    attention_correctness_score = dict()
    lrp_correctness_score = dict()
    count = 1
    num_test_sample = len(dataset.test_set)
    for X, y, data in dataset_provider.test_set(include_datum=True):
        img_filename = data[0].img_filename
        if img_filename not in category_dict.keys():
            continue
        print(count)
        if count > num_test_sample:
            break
        model = ImgCaptioningAdaptiveAttentionModel(COCO_config)
        model.build(dataset_provider.vocabs, dataset_provider.vocab_size)
        explainer = ExplainImgCaptioningAdaptiveAttention(model, model_weight_path, dataset_provider, max_caption_length)
        evaluation_engin = EvaluationBboxCOCO(category_dict=category_dict, max_caption_length=max_caption_length,
                                          beam_size=3, rule='eps', img_encode=cnn_type, explainer=explainer)
        lrp_score, attention_score, category_key = evaluation_engin.evaluate(X, data)
        for key in lrp_score.keys():
            if key not in lrp_correctness_score.keys():
                lrp_correctness_score[key] = dict()
                attention_correctness_score[key] = dict()
                lrp_correctness_score[key]['score'] = dict()
                attention_correctness_score[key]['score'] = dict()
                for threshold in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                    lrp_correctness_score[key]['score'][str(threshold)] = []
                    attention_correctness_score[key]['score'][str(threshold)]= []
                lrp_correctness_score[key]['category'] = category_key[key]
                attention_correctness_score[key]['category'] = category_key[key]
                lrp_correctness_score[key]['count'] = 0.
                attention_correctness_score[key]['count'] = 0.
            lrp_correctness_score[key]['count'] += 1
            attention_correctness_score[key]['count'] += 1
            for threshold in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                lrp_correctness_score[key]['score'][str(threshold)].append(lrp_score[key][str(threshold)])
                attention_correctness_score[key]['score'][str(threshold)].append(attention_score[key][str(threshold)])
        count += 1
        K.clear_session()
    save_root_path = './results/coco/evaluation-results/onelayer_VGG16_local_attentionv3/'
    save_lrp_name = 'coco_correctness_lrp_negative' + '.json'
    # save_attention_name = 'coco_correctness_attention' + '.json'
    with open(save_root_path + save_lrp_name, 'w') as f:
        json.dump(lrp_correctness_score,f)
        f.close()
    # with open(save_root_path + save_attention_name, 'w') as f:
    #     json.dump(attention_correctness_score, f)
    #     f.close()
def evaluate_coco_adaptive_InputTimesGradient(max_caption_length=20):

    category_path = './dataset/coco/COCOvalEntities.json'
    model_weight_path= './results/coco/training-results/coco_training_onelayer_VGG16_local_attentionv3_finetune3/keras_model_05_0.7816.hdf5'
    dataset = COCODataset(single_caption=True)
    COCO_config = config.COCOConfig()
    COCO_config.batch_size = 1
    dataset_provider = DatasetPreprocessorAttention(dataset, COCO_config, single_caption=True)


    with open(category_path, 'r') as f:
        category_dict = json.load(f)
        f.close()
    cnn_type = COCO_config.img_encoder

    gradient_correctness_score = dict()
    count = 1
    num_test_sample = len(dataset.test_set)
    for X, y, data in dataset_provider.test_set(include_datum=True):
        img_filename = data[0].img_filename
        if img_filename not in category_dict.keys():
            continue
        print(count)
        if count > num_test_sample:
            break
        model = ImgCaptioningAdaptiveAttentionModel(COCO_config)
        model.build(dataset_provider.vocabs, dataset_provider.vocab_size)
        explainer = ExplainImgCaptioningAdaptiveAttentionInputTimesGradient(model, model_weight_path, dataset_provider, max_caption_length)
        evaluation_engin = EvaluationBboxCOCOBaseline(category_dict=category_dict, max_caption_length=max_caption_length,
                                          beam_size=3, rule='eps', img_encode=cnn_type, explainer=explainer)
        gradient_score, category_key = evaluation_engin.evaluate(X, data)
        for key in gradient_score.keys():
            if key not in gradient_correctness_score.keys():
                gradient_correctness_score[key] = dict()
                gradient_correctness_score[key]['score'] = dict()
                for threshold in [0, 0.1, 0.2, 0.3, 0.4,0.5, 0.6, 0.7, 0.8, 0.9]:
                    gradient_correctness_score[key]['score'][str(threshold)] = []
                gradient_correctness_score[key]['category'] = category_key[key]
                gradient_correctness_score[key]['count'] = 0.
            gradient_correctness_score[key]['count'] += 1
            for threshold in [0, 0.1, 0.2, 0.3, 0.4, 0.5,0.6, 0.7, 0.8, 0.9]:
                gradient_correctness_score[key]['score'][str(threshold)].append(gradient_score[key][str(threshold)])
        count += 1

        K.clear_session()
    save_root_path = './results/coco/evaluation-results/without_norm/onelayer_VGG16_local_attentionv3/'
    save_gradient_name = 'coco_correctness_gradient' + '.json'
    with open(save_root_path + save_gradient_name, 'w') as f:
        json.dump(gradient_correctness_score,f)
        f.close()
def evaluate_coco_adaptive_guidedgradcam(max_caption_length=20):

    category_path = './dataset/coco/COCOvalEntities.json'
    model_weight_path= './results/coco/training-results/coco_training_onelayer_VGG16_local_attentionv3_finetune3/keras_model_05_0.7816.hdf5'
    dataset = COCODataset(single_caption=True)
    COCO_config = config.COCOConfig()
    COCO_config.batch_size = 1
    dataset_provider = DatasetPreprocessorAttention(dataset, COCO_config, single_caption=True)
    with open(category_path, 'r') as f:
        category_dict = json.load(f)
        f.close()
    cnn_type = COCO_config.img_encoder

    gradient_correctness_score = dict()
    count = 1
    num_test_sample = len(dataset.test_set)
    for X, y, data in dataset_provider.test_set(include_datum=True):
        img_filename = data[0].img_filename
        if img_filename not in category_dict.keys():
            continue
        if count > num_test_sample:
            break
        model = ImgCaptioningAdaptiveAttentionModel(COCO_config)
        model.build(dataset_provider.vocabs, dataset_provider.vocab_size)
        explainer = ExplainImgCaptioningAdaptiveAttentionGuidedGradcam(model, model_weight_path, dataset_provider, max_caption_length)
        evaluation_engin = EvaluationBboxCOCOBaseline(category_dict=category_dict, max_caption_length=max_caption_length,
                                          beam_size=3, rule='eps', img_encode=cnn_type, explainer=explainer)
        gradient_score, category_key = evaluation_engin.evaluate(X, data)
        for key in gradient_score.keys():
            if key not in gradient_correctness_score.keys():
                gradient_correctness_score[key] = dict()
                gradient_correctness_score[key]['score'] = dict()
                for threshold in [0, 0.1, 0.2, 0.3, 0.4,0.5, 0.6, 0.7, 0.8, 0.9]:
                    gradient_correctness_score[key]['score'][str(threshold)] = []
                gradient_correctness_score[key]['category'] = category_key[key]
                gradient_correctness_score[key]['count'] = 0.
            gradient_correctness_score[key]['count'] += 1
            for threshold in [0, 0.1, 0.2, 0.3, 0.4, 0.5,0.6, 0.7, 0.8, 0.9]:
                gradient_correctness_score[key]['score'][str(threshold)].append(gradient_score[key][str(threshold)])
        count += 1
        K.clear_session()
    save_root_path = './results/coco/evaluation-results/onelayer_VGG16_local_attentionv3/'
    save_gradient_name = 'coco_correctness_guidedgradcam_negative' + '.json'
    with open(save_root_path + save_gradient_name, 'w') as f:
        json.dump(gradient_correctness_score,f)
        f.close()


def evaluate_coco_gridTD(max_caption_length=20):
    category_path = './dataset/coco/COCOvalEntities.json'
    model_weight_path= './results/coco/training-results/coco_training_onelayer_VGG16_adaptive_bottomup_finetune2/keras_model_16_0.8171.hdf5'
    dataset = COCODataset(single_caption=True)
    COCO_config = config.COCOConfig()
    COCO_config.batch_size = 1
    dataset_provider = DatasetPreprocessorAttention(dataset, COCO_config, single_caption=True)

    with open(category_path, 'r') as f:
        category_dict = json.load(f)
        f.close()
    cnn_type = COCO_config.img_encoder
    attention_correctness_score = dict()
    lrp_correctness_score = dict()
    count = 1
    num_test_sample = len(dataset.test_set)
    for X, y, data in dataset_provider.test_set(include_datum=True):
        img_filename = data[0].img_filename
        if img_filename not in category_dict.keys():
            continue
        print(count)
        if count > num_test_sample:
            break
        model = ImgCaptioninggridTDAdaptiveModel(COCO_config)
        model.build(dataset_provider.vocabs, dataset_provider.vocab_size)
        explainer = ExplainImgCaptioningGridTDModel(model, model_weight_path, dataset_provider, max_caption_length)
        evaluation_engin = EvaluationBboxCOCO(category_dict=category_dict, max_caption_length=max_caption_length,
                                          beam_size=3, rule='eps', img_encode=cnn_type, explainer=explainer)
        lrp_score, attention_score, category_key = evaluation_engin.evaluate(X, data)
        for key in lrp_score.keys():
            if key not in lrp_correctness_score.keys():
                lrp_correctness_score[key] = dict()
                attention_correctness_score[key] = dict()
                lrp_correctness_score[key]['score'] = dict()
                attention_correctness_score[key]['score'] = dict()
                for threshold in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                    lrp_correctness_score[key]['score'][str(threshold)] = []
                    attention_correctness_score[key]['score'][str(threshold)]= []
                lrp_correctness_score[key]['category'] = category_key[key]
                attention_correctness_score[key]['category'] = category_key[key]
                lrp_correctness_score[key]['count'] = 0.
                attention_correctness_score[key]['count'] = 0.
            lrp_correctness_score[key]['count'] += 1
            attention_correctness_score[key]['count'] += 1
            for threshold in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                lrp_correctness_score[key]['score'][str(threshold)].append(lrp_score[key][str(threshold)])
                attention_correctness_score[key]['score'][str(threshold)].append(attention_score[key][str(threshold)])
        count += 1
        K.clear_session()
    save_root_path = './results/coco/evaluation-results/onelayer_VGG16_adaptive_bottomup/'
    save_lrp_name = 'coco_correctness_lrp_negative' + '.json'
    # save_attention_name = 'coco_correctness_attention' + '.json'
    with open(save_root_path + save_lrp_name, 'w') as f:
        json.dump(lrp_correctness_score,f)
        f.close()
    # with open(save_root_path + save_attention_name, 'w') as f:
    #     json.dump(attention_correctness_score, f)
    #     f.close()
def evaluate_coco_gridTD_InputTimesGradient(max_caption_length=20):

    category_path = './dataset/coco/COCOvalEntities.json'
    model_weight_path= './results/coco/training-results/coco_training_onelayer_VGG16_adaptive_bottomup_finetune2/keras_model_16_0.8171.hdf5'
    dataset = COCODataset(single_caption=True)
    COCO_config = config.COCOConfig()
    COCO_config.batch_size = 1
    dataset_provider = DatasetPreprocessorAttention(dataset, COCO_config, single_caption=True)
    with open(category_path, 'r') as f:
        category_dict = json.load(f)
        f.close()
    cnn_type = COCO_config.img_encoder

    gradient_correctness_score = dict()
    count = 1
    num_test_sample = len(dataset.test_set)
    for X, y, data in dataset_provider.test_set(include_datum=True):
        img_filename = data[0].img_filename
        if img_filename not in category_dict.keys():
            continue
        print(count)
        if count > num_test_sample:
            break
        model = ImgCaptioninggridTDAdaptiveModel(COCO_config)
        model.build(dataset_provider.vocabs, dataset_provider.vocab_size)
        explainer = ExplainImgCaptioningGridTDGradientTimesInput(model, model_weight_path, dataset_provider, max_caption_length)
        evaluation_engin = EvaluationBboxCOCOBaseline(category_dict=category_dict, max_caption_length=max_caption_length,
                                          beam_size=3, rule='eps', img_encode=cnn_type, explainer=explainer)
        gradient_score, category_key = evaluation_engin.evaluate(X, data)
        for key in gradient_score.keys():
            if key not in gradient_correctness_score.keys():
                gradient_correctness_score[key] = dict()
                gradient_correctness_score[key]['score'] = dict()
                for threshold in [0, 0.1, 0.2, 0.3, 0.4,0.5, 0.6, 0.7, 0.8, 0.9]:
                    gradient_correctness_score[key]['score'][str(threshold)] = []
                gradient_correctness_score[key]['category'] = category_key[key]
                gradient_correctness_score[key]['count'] = 0.
            gradient_correctness_score[key]['count'] += 1
            for threshold in [0, 0.1, 0.2, 0.3, 0.4, 0.5,0.6, 0.7, 0.8, 0.9]:
                gradient_correctness_score[key]['score'][str(threshold)].append(gradient_score[key][str(threshold)])
        count += 1

        K.clear_session()
    save_root_path = './results/coco/evaluation-results/without_norm/onelayer_VGG16_adaptive_bottomup/'
    save_gradient_name = 'coco_correctness_gradient' + '.json'
    with open(save_root_path + save_gradient_name, 'w') as f:
        json.dump(gradient_correctness_score,f)
        f.close()
def evaluate_coco_gridTD_guidedgradcam(max_caption_length=20):
    category_path = './dataset/coco/COCOvalEntities.json'
    model_weight_path = './results/coco/training-results/coco_training_onelayer_VGG16_adaptive_bottomup_finetune2/keras_model_16_0.8171.hdf5'
    dataset = COCODataset(single_caption=True)
    COCO_config = config.COCOConfig()
    COCO_config.batch_size = 1
    dataset_provider = DatasetPreprocessorAttention(dataset, COCO_config, single_caption=True)
    with open(category_path, 'r') as f:
        category_dict = json.load(f)
        f.close()
    cnn_type = COCO_config.img_encoder

    gradient_correctness_score = dict()
    count = 1
    num_test_sample = len(dataset.test_set)
    for X, y, data in dataset_provider.test_set(include_datum=True):
        img_filename = data[0].img_filename
        if img_filename not in category_dict.keys():
            continue
        print(count)
        if count > num_test_sample:
            break
        model = ImgCaptioninggridTDAdaptiveModel(COCO_config)
        model.build(dataset_provider.vocabs, dataset_provider.vocab_size)
        explainer = ExplainImgCaptioningGridTDGuidedGradcam(model, model_weight_path, dataset_provider,
                                                                           max_caption_length)
        evaluation_engin = EvaluationBboxCOCOBaseline(category_dict=category_dict,
                                                      max_caption_length=max_caption_length,
                                                      beam_size=3, rule='eps', img_encode=cnn_type, explainer=explainer)
        gradient_score, category_key = evaluation_engin.evaluate(X, data)
        for key in gradient_score.keys():
            if key not in gradient_correctness_score.keys():
                gradient_correctness_score[key] = dict()
                gradient_correctness_score[key]['score'] = dict()
                for threshold in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                    gradient_correctness_score[key]['score'][str(threshold)] = []
                gradient_correctness_score[key]['category'] = category_key[key]
                gradient_correctness_score[key]['count'] = 0.
            gradient_correctness_score[key]['count'] += 1
            for threshold in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                gradient_correctness_score[key]['score'][str(threshold)].append(gradient_score[key][str(threshold)])
        count += 1

        K.clear_session()
    save_root_path = './results/coco/evaluation-results/onelayer_VGG16_adaptive_bottomup/'
    save_gradient_name = 'coco_correctness_gradcam_negative' + '.json'
    with open(save_root_path + save_gradient_name, 'w') as f:
        json.dump(gradient_correctness_score, f)
        f.close()


def analyze_flickr_model(file_root_path):

    file_name_lrp = 'flickr30k_correctness_lrp' + '.json'
    file_name_attention = 'flickr30k_correctness_attention' + '.json'
    file_name_gradient = 'flickr30k_correctness_gradient' + '.json'
    file_name_guidedgradcam = 'flickr30k_correctness_guidedgradcam' + '.json'
    file_name_gradcam = 'flickr30k_correctness_gradcam' + '.json'
    with open(file_root_path + file_name_lrp, 'r') as f:
        lrp_dict = json.load(f)
        f.close()
    with open(file_root_path + file_name_attention, 'r') as f:
        attention_dict = json.load(f)
        f.close()
    with open(file_root_path + file_name_gradient, 'r') as f:
        gradient_dict = json.load(f)
        f.close()

    with open(file_root_path + file_name_gradcam, 'r') as f:
        gradcam_dict = json.load(f)
        f.close()

    with open(file_root_path + file_name_guidedgradcam, 'r') as f:
        guidedgradcam_dict = json.load(f)
        f.close()
    '''calculate the average correctness'''
    save_data = np.zeros((11,10))
    lrp_score = dict()
    lrp_count = 0.
    attention_score = dict()
    attention_count = 0.
    gradient_score = dict()
    gradient_count = 0.
    gradcam_score = dict()
    gradcam_count = 0.
    guidedgradcam_score = dict()
    guidedgradcam_count = 0.
    whole_lrp_score = dict()
    whole_attention_score = dict()
    whole_gradient_score = dict()
    whole_gradcam_score = dict()
    whole_guidedgradcam_score = dict()
    for key in lrp_dict.keys():
        for threshold in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            if str(threshold) not in lrp_score.keys():
                lrp_score[str(threshold)] = 0.
                attention_score[str(threshold)] = 0.
                gradient_score[str(threshold)] = 0.
                gradcam_score[str(threshold)] = 0.
                guidedgradcam_score[str(threshold)] = 0.
                whole_lrp_score[str(threshold)] = []
                whole_attention_score[str(threshold)] = []
                whole_gradient_score[str(threshold)] = []
                whole_gradcam_score[str(threshold)] = []
                whole_guidedgradcam_score[str(threshold)] = []
            lrp_score[str(threshold)] += np.array(lrp_dict[key]['score'][str(threshold)]).sum()
            attention_score[str(threshold)] += np.array(attention_dict[key]['score'][str(threshold)]).sum()
            gradient_score[str(threshold)] += np.array(gradient_dict[key]['score'][str(threshold)]).sum()
            gradcam_score[str(threshold)] += np.array(gradcam_dict[key]['score'][str(threshold)]).sum()
            guidedgradcam_score[str(threshold)] += np.array(guidedgradcam_dict[key]['score'][str(threshold)]).sum()
            whole_lrp_score[str(threshold)] += lrp_dict[key]['score'][str(threshold)]
            whole_gradient_score[str(threshold)] += gradient_dict[key]['score'][str(threshold)]
            whole_attention_score[str(threshold)] += attention_dict[key]['score'][str(threshold)]
            whole_gradcam_score[str(threshold)] += gradcam_dict[key]['score'][str(threshold)]
            whole_guidedgradcam_score[str(threshold)] += guidedgradcam_dict[key]['score'][str(threshold)]
        lrp_count += lrp_dict[key]['count']
        attention_count += attention_dict[key]['count']
        gradient_count += gradient_dict[key]['count']
        gradcam_count += gradcam_dict[key]['count']
        guidedgradcam_count += guidedgradcam_dict[key]['count']
    print('total count', lrp_count)
    idx = 0
    for threshold in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        save_data[0, idx] = threshold
        lrp_score[str(threshold)] = lrp_score[str(threshold)] / lrp_count
        attention_score[str(threshold)] = attention_score[str(threshold)] / attention_count
        gradient_score[str(threshold)] = gradient_score[str(threshold)] / gradient_count
        gradcam_score[str(threshold)] = gradcam_score[str(threshold)] / gradcam_count
        guidedgradcam_score[str(threshold)] =guidedgradcam_score[str(threshold)] / guidedgradcam_count
        save_data[1, idx] = lrp_score[str(threshold)]
        save_data[2, idx] = np.std(np.array(whole_lrp_score[str(threshold)]))
        save_data[3, idx] = attention_score[str(threshold)]
        save_data[4, idx] = np.std(np.array(whole_attention_score[str(threshold)]))
        save_data[5, idx] = gradient_score[str(threshold)]
        save_data[6, idx] = np.std(np.array(whole_gradient_score[str(threshold)]))
        save_data[7, idx] = gradcam_score[str(threshold)]
        save_data[8, idx] = np.std(np.array(whole_gradcam_score[str(threshold)]))
        save_data[9, idx] = guidedgradcam_score[str(threshold)]
        save_data[10, idx] = np.std(np.array(whole_guidedgradcam_score[str(threshold)]))

        print(threshold, lrp_score[str(threshold)])
        print(threshold, attention_score[str(threshold)])
        idx+=1

    np.savetxt(file_root_path + 'flickr30k_analized_data.csv', save_data, delimiter=',')

    '''write CSV'''
    category_dict = dict()

    for key in lrp_dict.keys():

        if key not in category_dict.keys():
            category_dict[key] = dict()
        for item in Category_FILTER:
            if item in lrp_dict[key]['category']:
                lrp_dict[key]['category'] = lrp_dict[key]['category'].lstrip(item)
                lrp_dict[key]['category'] = lrp_dict[key]['category'].strip()
        category_dict[key]['category'] = lrp_dict[key]['category'].lower()
        category_dict[key]['lrp'] = 1.0 * np.array(lrp_dict[key]['score']['0']).sum() / lrp_dict[key]['count']
        category_dict[key]['attention'] = 1.0 * np.array(attention_dict[key]['score']['0']).sum() / attention_dict[key]['count']
        category_dict[key]['gradient*input'] = 1.0 * np.array(gradient_dict[key]['score']['0']).sum() / gradient_dict[key]['count']
        category_dict[key]['lrp_std'] = np.std(np.array(lrp_dict[key]['score']['0']))
        category_dict[key]['attention_std'] = np.std(np.array(attention_dict[key]['score']['0']))
        category_dict[key]['gradient*input_std'] = np.std(np.array(gradient_dict[key]['score']['0']))
    merged_category_dict = dict()
    for key in category_dict.keys():
        category = category_dict[key]['category']
        if category not in merged_category_dict.keys():
            merged_category_dict[category] = dict()
            merged_category_dict[category]['category'] = category
            merged_category_dict[category]['lrp'] = []
            merged_category_dict[category]['attention'] = []
            merged_category_dict[category]['gradient*input'] = []
        merged_category_dict[category]['lrp'].append(category_dict[key]['lrp'])
        merged_category_dict[category]['attention'].append(category_dict[key]['attention'])
        merged_category_dict[category]['gradient*input'].append(category_dict[key]['gradient*input'])
    for key in merged_category_dict.keys():
        if len(merged_category_dict[key]['lrp']) > 1:
            merged_category_dict[key]['lrp_std'] = np.std(np.array(merged_category_dict[key]['lrp']))
            merged_category_dict[key]['attention_std'] = np.std(np.array(merged_category_dict[key]['attention']))
            merged_category_dict[key]['gradient*input_std'] = np.std(np.array(merged_category_dict[key]['gradient*input']))
        else:
            merged_category_dict[key]['lrp_std'] = 0
            merged_category_dict[key]['attention_std'] = 0
            merged_category_dict[key]['gradient*input_std'] = 0
        merged_category_dict[key]['lrp'] = np.mean(np.array(merged_category_dict[key]['lrp']))
        merged_category_dict[key]['attention'] = np.mean(np.array(merged_category_dict[key]['attention']))
        merged_category_dict[key]['gradient*input'] = np.mean(np.array(merged_category_dict[key]['gradient*input']))


    rows = []
    header = ['category', 'lrp', 'lrp_std', 'attention','attention_std', 'gradient*input','gradient*input_std']
    for key in merged_category_dict.keys():
        rows.append(merged_category_dict[key])

    with open(file_root_path + 'flickr30k_category_scores_0.csv', 'w') as f:
        f_csv = csv.DictWriter(f, header)
        f_csv.writeheader()
        f_csv.writerows(rows)
        f.close()

def analyze_coco_model(file_root_path):

    file_name_lrp = 'coco_correctness_lrp_negative' + '.json'
    file_name_attention = 'coco_correctness_attention' + '.json'
    file_name_gradient = 'coco_correctness_gradient' + '.json'
    file_name_guidedgradcam = 'coco_correctness_guidedgradcam_negative' + '.json'
    file_name_gradcam = 'coco_correctness_gradcam' + '.json'


    with open(file_root_path + file_name_lrp, 'r') as f:
        lrp_dict = json.load(f)
        f.close()
    with open(file_root_path + file_name_attention, 'r') as f:
        attention_dict = json.load(f)
        f.close()
    with open(file_root_path + file_name_gradient, 'r') as f:
        gradient_dict = json.load(f)
        f.close()

    with open(file_root_path + file_name_gradcam, 'r') as f:
        gradcam_dict = json.load(f)
        f.close()

    with open(file_root_path + file_name_guidedgradcam, 'r') as f:
        guidedgradcam_dict = json.load(f)
        f.close()

    '''calculate the average correctness'''
    save_data = np.zeros((11,10))
    lrp_score = dict()
    lrp_count = 0.
    attention_score = dict()
    attention_count = 0.
    gradient_score = dict()
    gradient_count = 0.
    gradcam_score = dict()
    gradcam_count = 0.
    guidedgradcam_score = dict()
    guidedgradcam_count = 0.
    whole_lrp_score = dict()
    whole_attention_score = dict()
    whole_gradient_score = dict()
    whole_gradcam_score = dict()
    whole_guidedgradcam_score = dict()
    for key in lrp_dict.keys():
        for threshold in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            if str(threshold) not in lrp_score.keys():
                lrp_score[str(threshold)] = 0.
                attention_score[str(threshold)] = 0.
                gradient_score[str(threshold)] = 0.
                gradcam_score[str(threshold)] = 0.
                guidedgradcam_score[str(threshold)] = 0.
                whole_lrp_score[str(threshold)] = []
                whole_attention_score[str(threshold)] = []
                whole_gradient_score[str(threshold)] = []
                whole_gradcam_score[str(threshold)] = []
                whole_guidedgradcam_score[str(threshold)] = []
            lrp_score[str(threshold)] += np.array(lrp_dict[key]['score'][str(threshold)]).sum()
            attention_score[str(threshold)] += np.array(attention_dict[key]['score'][str(threshold)]).sum()
            gradient_score[str(threshold)] += np.array(gradient_dict[key]['score'][str(threshold)]).sum()
            gradcam_score[str(threshold)] += np.array(gradcam_dict[key]['score'][str(threshold)]).sum()
            guidedgradcam_score[str(threshold)] += np.array(guidedgradcam_dict[key]['score'][str(threshold)]).sum()
            whole_lrp_score[str(threshold)] += lrp_dict[key]['score'][str(threshold)]
            whole_gradient_score[str(threshold)] += gradient_dict[key]['score'][str(threshold)]
            whole_attention_score[str(threshold)] += attention_dict[key]['score'][str(threshold)]
            whole_gradcam_score[str(threshold)] += gradcam_dict[key]['score'][str(threshold)]
            whole_guidedgradcam_score[str(threshold)] += guidedgradcam_dict[key]['score'][str(threshold)]
        lrp_count += lrp_dict[key]['count']
        attention_count += attention_dict[key]['count']
        gradient_count += gradient_dict[key]['count']
        gradcam_count += gradcam_dict[key]['count']
        guidedgradcam_count += guidedgradcam_dict[key]['count']
    print('total count', lrp_count)
    idx = 0
    for threshold in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        save_data[0, idx] = threshold
        lrp_score[str(threshold)] = lrp_score[str(threshold)] / lrp_count
        attention_score[str(threshold)] = attention_score[str(threshold)] / attention_count
        gradient_score[str(threshold)] = gradient_score[str(threshold)] / gradient_count
        gradcam_score[str(threshold)] = gradcam_score[str(threshold)] / gradcam_count
        guidedgradcam_score[str(threshold)] =guidedgradcam_score[str(threshold)] / guidedgradcam_count
        save_data[1, idx] = lrp_score[str(threshold)]
        save_data[2, idx] = np.std(np.array(whole_lrp_score[str(threshold)]))
        save_data[3, idx] = attention_score[str(threshold)]
        save_data[4, idx] = np.std(np.array(whole_attention_score[str(threshold)]))
        save_data[5, idx] = gradient_score[str(threshold)]
        save_data[6, idx] = np.std(np.array(whole_gradient_score[str(threshold)]))
        save_data[7, idx] = gradcam_score[str(threshold)]
        save_data[8, idx] = np.std(np.array(whole_gradcam_score[str(threshold)]))
        save_data[9, idx] = guidedgradcam_score[str(threshold)]
        save_data[10, idx] = np.std(np.array(whole_guidedgradcam_score[str(threshold)]))

        print(threshold, lrp_score[str(threshold)])
        print(threshold, attention_score[str(threshold)])
        idx+=1

    np.savetxt(file_root_path + 'coco_analized_data_negative.csv', save_data, delimiter=',')



if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # evaluate_coco_gridTD()
    # evaluate_coco_gridTD_InputTimesGradient()
    # evaluate_coco_gridTD_guidedgradcam()

    evaluate_coco_adaptive(20)
    # evaluate_coco_adaptive_InputTimesGradient()
    # evaluate_coco_adaptive_guidedgradcam()


    # file_root_path = './results/flickr30k/evaluation-results/onelayer_VGG16_local_attentionv3/'
    # file_root_path = './results/flickr30k/evaluation-results/onelayer_VGG16_adaptive_bottomup/'
    # file_root_path = './results/flickr30k/evaluation-results/onelayer_VGG16_show_attend/'
    # analyze_flickr_model(file_root_path)
    #
    # file_root_path = './results/coco/evaluation-results/onelayer_VGG16_show_attend/'
    # file_root_path = './results/coco/evaluation-results/onelayer_VGG16_adaptive_bottomup/'
    # file_root_path = './results/coco/evaluation-results/onelayer_VGG16_local_attentionv3/'
    # analyze_coco_model(file_root_path)











