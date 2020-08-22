from models.explainers import *


class Explainer(object):

    def __init__(self,model, weight_path, explainer, max_caption_length, beam_size):
        self._keras_model = model.keras_model
        self._keras_model.load_weights(weight_path)
        self._image_model = Model(inputs=self._keras_model.get_layer('input_1').input,
                                  outputs=self._keras_model.get_layer('block5_conv3').output)
        self._img_encoder = model.img_encoder
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
    def _project(self, x):
        absmax = np.max(np.abs(x))
        x = 1.0 * x/absmax
        if np.sum(x<0):
            x = (x + 1)/2
        else:
            x = x
        return x * 255
    def _preprocess_img(self, img_path):
        preprocessed_img = self._image_preprocessor.preprocess_images(img_path)
        img_array = self._image_preprocessor.preprocess_batch(preprocessed_img)
        initial_caption = self._caption_preprocessor.SOS_TOKEN_LABEL_ENCODED
        return (initial_caption, img_array)

    def _predict_caption(self, X):
        captions = self._explainer._beam_search(X, beam_size=self._beam_size)[0]
        return captions

    def _explain_captions(self, X, captions, save_folder):
        self._explainer._forward_beam_search(X, captions)
        beam_search_captions = self._caption_preprocessor.decode_captions_from_list1d(captions)
        print(beam_search_captions)
        img_encode_relevance, attention = self._explainer._explain_sentence()
        sequence_input, img_input = X
        x = int(np.sqrt(len(attention)))
        y = int(np.ceil(len(attention) / x))
        _, axes = plt.subplots(y, x, sharex="col", sharey="row", figsize=(20, 20))
        axes = axes.flatten()
        for i in range(len(attention)):
            relevance = self._explainer._explain_CNN(img_input, img_encode_relevance[i])
            channels_first = K.image_data_format() == "channels_first"
            hp = postprocess(relevance, self._color_conversion, channels_first)
            hp = heatmap(hp)
            axes[i].set_title(self._caption_preprocessor._word_of[captions[i]], fontsize=18)
            axes[i].imshow(hp[0])
        plt.savefig(os.path.join(save_folder, 'lrp_hm.jpg'))
        x = int(np.sqrt(len(attention)))
        y = int(np.ceil(len(attention) / x))
        _, axes = plt.subplots(y, x, sharex="col", sharey="row", figsize=(20, 20))
        plt.axis('off')
        axes = axes.flatten()
        img_original = Image.open(self.img_path)
        blank = np.zeros((224, 224, 2))
        img_original = img_original.resize((224, 224))
        img_original.save(os.path.join(save_folder, save_folder.split('/')[-1]))
        for i in range(len(attention)):
            if self._img_encoder == "inception_v3":
                atn = skimage.transform.pyramid_expand(attention[i].reshape(5,5), upscale=20, sigma=20,
                                                       multichannel=False)
                atn = skimage.transform.resize(atn, (224, 224), mode='reflect', anti_aliasing=True)
            else:
                atn = skimage.transform.pyramid_expand(attention[i].reshape(14, 14), upscale=16, sigma=20,
                                                       multichannel=False)
            atn = atn[:, :, np.newaxis]
            atn = (atn - np.min(atn)) / np.max(atn) * 255
            atn = np.concatenate((atn, blank), axis=-1)
            attention_img = Image.fromarray(np.uint8(atn), img_original.mode)
            tmp_img = Image.blend(img_original, attention_img, 0.7)
            axes[i].set_title(self._caption_preprocessor._word_of[captions[i]], fontsize=18)
            axes[i].imshow(tmp_img)
        plt.savefig(os.path.join(save_folder, 'attention.jpg'))

    def _explain_color_word(self, X, captions, save_folder, t):
        img_original = Image.open(self.img_path)
        img_original = img_original.resize((224, 224))
        img_original.save(os.path.join(save_folder, save_folder.split('/')[-1]))
        img_red, img_green, img_blue = img_original.split()
        img_red.save(os.path.join(save_folder, 'R_' + save_folder.split('/')[-1]))
        img_green.save(os.path.join(save_folder, 'G_' + save_folder.split('/')[-1]))
        img_blue.save(os.path.join(save_folder, 'B_' + save_folder.split('/')[-1]))
        img_original.show()
        self._explainer._forward_beam_search(X, captions)
        sequence_input, img_input = X
        img_encode_relevance, attention = self._explainer._explain_lstm_single_word_sequence(t)
        relevance = self._explainer._explain_CNN(img_input, img_encode_relevance)
        channels_first = K.image_data_format() == "channels_first"
        hp_3channels = postprocess(relevance, self._color_conversion, channels_first)[0]
        for channel in ['R', 'G', 'B']:
            index = ['R', 'G', 'B'].index(channel)
            hp = hp_3channels[:,:,index]
            hp = hp[np.newaxis, :, :, np.newaxis]
            hp = np.maximum(hp, 0)
            hp = heatmap(hp)[0]
            hp = self._project(hp)
            hp_img = Image.fromarray(np.uint8(hp))
            hp_img.show()
            hp_img.save(os.path.join(save_folder, 'lrp_' + channel + '_' + save_folder.split('/')[-1].split('.')[0]+ '_' + self._caption_preprocessor._word_of[captions[t-1]]+ '.png'))
        atn = skimage.transform.pyramid_expand(attention.reshape(self._reshape_size), upscale=self._upscale, sigma=20,  multichannel=False)
        atn = atn[:, :, np.newaxis]
        atn = self._project(atn)
        blank = np.zeros((224, 224, 2))
        atn = np.concatenate((atn, blank), axis=-1)
        attention_img = Image.fromarray(np.uint8(atn), img_original.mode)
        tmp_img = Image.blend(img_original, attention_img, 0.7)
        tmp_img.save(os.path.join(save_folder, 'attention_'+ save_folder.split('/')[-1].split('.')[0]+ '_' + self._caption_preprocessor._word_of[captions[t-1]]+ '.png'))

    def _explain_single_word(self, X, captions, save_folder, t):
        img_original = Image.open(self.img_path)
        img_original = img_original.resize((224, 224))
        img_original.save(os.path.join(save_folder, save_folder.split('/')[-1]))
        img_original.show()
        self._explainer._forward_beam_search(X, captions)
        sequence_input, img_input = X
        img_encode_relevance, attention = self._explainer._explain_lstm_single_word_sequence(t)
        relevance = self._explainer._explain_CNN(img_input, img_encode_relevance)
        channels_first = K.image_data_format() == "channels_first"
        hp= postprocess(relevance, self._color_conversion, channels_first)
        hp = heatmap(hp)[0]
        hp = self._project(hp)
        hp_img = Image.fromarray(np.uint8(hp))
        hp_img.show()
        hp_img.save(os.path.join(save_folder, 'lrp_' + save_folder.split('/')[-1].split('.')[0]+ '_' + self._caption_preprocessor._word_of[captions[t-1]]+ '.jpg'))
        atn = skimage.transform.pyramid_expand(attention.reshape(self._reshape_size), upscale=self._upscale, sigma=20,  multichannel=False)
        atn = atn[:, :, np.newaxis]
        atn = self._project(atn)
        blank = np.zeros((224, 224, 2))
        atn = np.concatenate((atn, blank), axis=-1)
        attention_img = Image.fromarray(np.uint8(atn), img_original.mode)
        tmp_img = Image.blend(img_original, attention_img, 0.7)
        tmp_img.save(os.path.join(save_folder, 'attention_'+ save_folder.split('/')[-1].split('.')[0]+ '_' + self._caption_preprocessor._word_of[captions[t-1]]+ '.jpg'))
        print(self._caption_preprocessor._word_of[captions[t]])
        print('start', self._explainer.r_words[0])
        for i in range(t-1):
            print(self._caption_preprocessor._word_of[captions[i]], self._explainer.r_words[i])

    def analyze_img(self, folder, img_path):
        self.img_path = img_path
        X = self._preprocess_img([img_path])
        captions = self._predict_caption(X)
        img_name = img_path.split('/')[-1]
        print(img_name)
        save_folder = os.path.join(folder, img_name)
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)
        self._explain_captions(X, captions, save_folder)

    def analyze_img_color(self,folder, img_path, t):
        self.img_path = img_path
        X = self._preprocess_img([img_path])
        captions = self._predict_caption(X)
        img_name = img_path.split('/')[-1]
        print(img_name)
        save_folder = os.path.join(folder, img_name)
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)
        self._explain_color_word(X, captions, save_folder, t)

    def analyze_single_word(self,folder, img_path, t):
        self.img_path = img_path
        X = self._preprocess_img([img_path])
        captions = self._predict_caption(X)
        img_name = img_path.split('/')[-1]
        print(img_name)
        save_folder = os.path.join(folder, img_name)
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)
        self._explain_single_word(X, captions, save_folder, t)


class ExplainerGuidedgradcam(Explainer):
    def __init__(self, model, weight_path, explainer, max_caption_length, beam_size):
        super(ExplainerGuidedgradcam, self).__init__(model, weight_path, explainer, max_caption_length, beam_size)

    def _explain_captions(self, X, captions, save_folder):
        self._explainer._forward_beam_search(X, captions)
        beam_search_captions = self._caption_preprocessor.decode_captions_from_list1d(captions)
        print(beam_search_captions)
        img_encode_relevance = self._explainer._explain_sentence()
        sequence_input, img_input = X
        x = int(np.sqrt(len(img_encode_relevance)))
        y = int(np.ceil(len(img_encode_relevance) / x))
        _, axes = plt.subplots(y, x, sharex="col", sharey="row", figsize=(20, 20))
        axes = axes.flatten()
        for i in range(len(img_encode_relevance)):
            relevance = self._explainer._explain_CNN(img_input, img_encode_relevance[i])
            channels_first = K.image_data_format() == "channels_first"
            hp = postprocess(relevance, self._color_conversion, channels_first)
            hp = heatmap(hp)
            axes[i].set_title(self._caption_preprocessor._word_of[captions[i]], fontsize=18)
            axes[i].imshow(hp[0])
        plt.savefig(os.path.join(save_folder, 'gradcam_hm.jpg'))

    def _explain_single_word(self, X, captions, save_folder, t, type='guidedgradcam'):
        def project(x):
            absmax = np.max(np.abs(x))
            x = 1.0 * x/absmax
            if np.sum(x<0):
                x = (x + 1)/2
            else:
                x = x
            return x * 255
        img_original = Image.open(self.img_path)
        img_original = img_original.resize((224, 224))
        img_original.save(os.path.join(save_folder, save_folder.split('/')[-1]))
        img_original.show()
        self._explainer._forward_beam_search(X, captions)
        beam_search_captions = self._caption_preprocessor.decode_captions_from_list1d(captions)
        sequence_input, img_input = X
        img_encode_relevance = self._explainer._lstm_decoder_backward(t)
        relevance = self._explainer._explain_CNN(img_input, img_encode_relevance)
        print(relevance.shape)
        if type=='gradcam':
            blank = np.zeros((224, 224, 2))
            relevance = relevance[0]
            relevance = relevance[:, :, 0:1]
            relevance = project(relevance)
            cam = np.concatenate((relevance, blank), axis=-1)
            cam_img = Image.fromarray(np.uint8(cam), img_original.mode)
            cam_img = Image.blend(img_original, cam_img, 0.7)
            cam_img.save(os.path.join(save_folder, 'gradcam_' + save_folder.split('/')[-1].split('.')[0] + '_' +
                                      self._caption_preprocessor._word_of[captions[t - 1]] + '.jpg'))
            for i in range(t - 1):
                print(self._caption_preprocessor._word_of[captions[i]], self._explainer.r_words[i + 1])
            return
        channels_first = K.image_data_format() == "channels_first"
        hp= postprocess(relevance, self._color_conversion, channels_first)
        hp = heatmap(hp)[0]
        # print(hp.shape)
        hp = project(hp)
        hp_img = Image.fromarray(np.uint8(hp))
            # hp_img = Image.blend(img_original,hp_img, 0.7)
        hp_img.show()
        hp_img.save(os.path.join(save_folder, 'guidedgradcam_' + save_folder.split('/')[-1].split('.')[0]+ '_' + self._caption_preprocessor._word_of[captions[t-1]]+ '.jpg'))
        print(self._caption_preprocessor._word_of[captions[t]])
        print('start', self._explainer.r_words[0])
        for i in range(t-1):
            print(self._caption_preprocessor._word_of[captions[i]], self._explainer.r_words[i+1])

    def analyze_single_word(self,folder, img_path, t, type='guidedgradcam'):
        self.img_path = img_path
        X = self._preprocess_img([img_path])
        captions = self._predict_caption(X)
        img_name = img_path.split('/')[-1]
        print(img_name)
        save_folder = os.path.join(folder, img_name)
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)
        self._explain_single_word(X, captions, save_folder, t, type=type)


def explain_COCOmodel(model_weight_path, model_type, explain_img_file, explainer_type, explain_single_word=False, single_word_idx=None, max_caption_length=20):
    '''
    This function explain the models trained on MSCOCO dataset
    :param model_weight_path: The hdf5 file of the model weight
    :param model_type: adaptive or gridTD
    :param explain_img_file: the path of a test image
    :param explainer_type: explanation methods, lrp or guidedgradcam,  in you want to test other methods, add an option and specify the explainer_engine
    :param explain_single_word: if true, this function only explains one word in the caption and will save the heatmap and print the explanation scores of the proceeding words.
    :param single_word_idx: only valid if the 'explain_single_word' is True. Specify the index of the target word, index starts from 1
    :param max_caption_length:
    :return: The explanation heatmaps
    '''
    dataset = COCODataset(single_caption=True)
    coco_config = config.COCOConfig()
    coco_config.batch_size = 1
    dataset_provider = DatasetPreprocessorAttention(dataset, coco_config, single_caption=True)
    if model_type == 'adaptive':
        model = ImgCaptioningAdaptiveAttentionModel(coco_config)
        model.build(dataset_provider.vocabs, dataset_provider.vocab_size)
        if explainer_type == 'lrp':
            explainer_engine = ExplainImgCaptioningAdaptiveAttention(model, model_weight_path, dataset_provider,
                                                                     max_caption_length)
            explainer = Explainer(model, model_weight_path, explainer_engine, max_caption_length, beam_size=3)
        elif explainer_type == 'guidedgradcam':
            explainer_engine = ExplainImgCaptioningAdaptiveAttentionGuidedGradcam(model, model_weight_path, dataset_provider,
                                                                                  max_caption_length)
            explainer = ExplainerGuidedgradcam(model, model_weight_path, explainer_engine, max_caption_length,
                                               beam_size=3)
        else:
            raise NotImplementedError('Please specify the explainer_type as lrp or guidedgradcam')
        save_folder = './example_images/coco_adaptive'
    elif model_type == 'gridTD':
        model = ImgCaptioninggridTDAdaptiveModel(coco_config)
        model.build(dataset_provider.vocabs, dataset_provider.vocab_size)
        if explainer_type == 'lrp':
            explainer_engine = ExplainImgCaptioningGridTDModel(model, model_weight_path, dataset_provider,
                                                               max_caption_length)
            explainer = Explainer(model, model_weight_path, explainer_engine, max_caption_length, beam_size=3)
        elif explainer_type == 'guidedgradcam':
            explainer_engine = ExplainImgCaptioningGridTDGuidedGradcam(model, model_weight_path, dataset_provider,
                                                                       max_caption_length)
            explainer = ExplainerGuidedgradcam(model, model_weight_path, explainer_engine, max_caption_length,
                                               beam_size=3)
        else:
            raise NotImplementedError('Please specify the explainer_type as lrp or guidedgradcam')
        save_folder = './example_images/coco_gridTD'
    else:
        raise NotImplementedError('Please specify the model_type as adaptive or gridTD')
    if explain_single_word:
        explainer.analyze_single_word(save_folder, explain_img_file, single_word_idx)
    else:
        explainer.analyze_img(save_folder, explain_img_file)


def explain_flickr30Kmodel(model_weight_path, model_type, explain_img_file, explainer_type, explain_single_word=False, single_word_idx=None, max_caption_length=20):
    '''
    This function explain the models trained on Flickr30k dataset
    :param model_weight_path: The hdf5 file of the model weight
    :param model_type: adaptive or gridTD
    :param explain_img_file: the path of a test image
    :param explainer_type: explanation methods, lrp or guidedgradcam,  in you want to test other methods, add an option and specify the explainer_engine
    :param explain_single_word: if true, this function only explains one word in the caption and will save the heatmap and print the explanation scores of the proceeding words.
    :param single_word_idx: only valid if the 'explain_single_word' is True. Specify the index of the target word, index starts from 1
    :param max_caption_length:
    :return: The explanation heatmaps
    '''
    flickr30_config = config.FlickrConfig()
    dataset = Flickr30kDataset(flickr30_config, single_caption=True)
    dataset_provider = DatasetPreprocessorAttention(dataset, flickr30_config, single_caption=True)
    if model_type == 'adaptive':
        model = ImgCaptioningAdaptiveAttentionModel(flickr30_config)
        model.build(dataset_provider.vocabs, dataset_provider.vocab_size)
        if explainer_type == 'lrp':
            explainer_engine = ExplainImgCaptioningAdaptiveAttention(model, model_weight_path, dataset_provider,
                                                                     max_caption_length)
            explainer = Explainer(model, model_weight_path, explainer_engine, max_caption_length, beam_size=3)
        elif explainer_type == 'guidedgradcam':
            explainer_engine = ExplainImgCaptioningAdaptiveAttentionGuidedGradcam(model, model_weight_path, dataset_provider,
                                                                                  max_caption_length)
            explainer = ExplainerGuidedgradcam(model, model_weight_path, explainer_engine, max_caption_length,
                                               beam_size=3)
        else:
            raise NotImplementedError('Please specify the explainer_type as lrp or guidedgradcam')
        save_folder = './example_images/flickr30k_adaptive'
    elif model_type == 'gridTD':
        model = ImgCaptioninggridTDAdaptiveModel(flickr30_config)
        model.build(dataset_provider.vocabs, dataset_provider.vocab_size)
        if explainer_type == 'lrp':
            explainer_engine = ExplainImgCaptioningGridTDModel(model, model_weight_path, dataset_provider,
                                                               max_caption_length)
            explainer = Explainer(model, model_weight_path, explainer_engine, max_caption_length, beam_size=3)
        elif explainer_type == 'guidedgradcam':
            explainer_engine = ExplainImgCaptioningGridTDGuidedGradcam(model, model_weight_path, dataset_provider,
                                                                       max_caption_length)
            explainer = ExplainerGuidedgradcam(model, model_weight_path, explainer_engine, max_caption_length,
                                               beam_size=3)
        else:
            raise NotImplementedError('Please specify the explainer_type as lrp or guidedgradcam')
        save_folder = './example_images/flickr30k_gridTD'
    else:
        raise NotImplementedError('Please specify the model_type as adaptive or gridTD')
    if explain_single_word:
        explainer.analyze_single_word(save_folder, explain_img_file, single_word_idx)
    else:
        explainer.analyze_img(save_folder, explain_img_file)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # model_weight_path = './results/flickr30k/training-results/flickr_VGG16_adaptive_attention/keras_model.hdf5'
    # model_weight_path = './results/flickr30k/training-results/flickr_VGG16_gridTD_attention/keras_model.hdf5'
    # model_weight_path = './results/coco/training-results/coco_VGG16_adaptive_attention/keras_model.hdf5'
    model_weight_path = './results/coco/training-results/coco_VGG16_gridTD_attention/keras_model.hdf5'
    model_type = 'gridTD'
    explainer_type = 'lrp'
    explain_img_file = './example_images/cocoimage/000000005586.jpg'
    explain_COCOmodel(model_weight_path, model_type, explain_img_file, explainer_type)
    # explain_flickr30Kmodel(model_weight_path, model_type, explain_img_file, explainer_type)



