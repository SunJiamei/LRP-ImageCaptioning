import numpy as np

from functools import partial
from keras.applications import inception_v3, vgg16, vgg19, resnet50
from keras.preprocessing import sequence as keras_seq
from keras.preprocessing.image import (ImageDataGenerator, img_to_array,
                                       load_img)
from keras.preprocessing.text import Tokenizer, text_to_word_sequence

class ImagePreprocessor(object):
    """A Inception v3 image preprocessor. Implements an image augmentation
    as well."""
    # IMAGE_SIZE = (299, 299) # Inception v3
    IMAGE_SIZE = (224, 224)  # VGG16

    def __init__(self, encoder, image_augmentation=False):

        self._image_data_generator = ImageDataGenerator(rotation_range=40,
                                                        width_shift_range=0.2,
                                                        height_shift_range=0.2,
                                                        shear_range=0.2,
                                                        zoom_range=0.2,
                                                        horizontal_flip=True,
                                                        fill_mode='nearest')

        self._image_augmentation_switch = image_augmentation
        self.encoder = encoder

    def preprocess_images(self, img_paths, random_transform=False):
        images = map(partial(self._preprocess_an_image,
                           random_transform=random_transform),
                   img_paths)     #return an iterable functon
        return list(images)

    def preprocess_batch(self, img_list):
        return np.array(img_list)

    def _preprocess_an_image(self, img_path, random_transform=True):
        img = load_img(img_path, target_size=self.IMAGE_SIZE)
        img_array = img_to_array(img)  # transform image to numpy array
        if self._image_augmentation_switch and random_transform:
            img_array = self._image_data_generator.random_transform(img_array)
        if self.encoder == 'vgg16':
            img_array = vgg16.preprocess_input(img_array)
        elif self.encoder == 'vgg19':
            img_array = vgg19.preprocess_input(img_array)
        elif self.encoder == 'inception_v3':
            img_array = inception_v3.preprocess_input(img_array)
        elif self.encoder == 'resnet101':
            img_array = resnet50.preprocess_input(img_array)
        else:
            raise NotImplementedError('do not have this encoder option')
        return img_array



class CaptionPreprocessorAttention(object):
    """Preprocesses captions before feeded into the network."""
    EOS_TOKEN = 'zeros'
    SOS_TOKEN = 'szeros'

    def __init__(self, datasetname, config):
        """
        If an arg is None, it will get its value from config.active_config.
        Args:
          rare_words_handling: {'nothing'|'discard'|'change'}
          words_min_occur: words whose occurrences are less than this are
                           considered rare words
        """
        self.config = config
        self._tokenizer = Tokenizer()
        self._rare_words_handling = self.config.rare_words_handling
        self._words_min_occur = self.config.words_min_occur
        self._word_of = {}
        self._datasetname = datasetname

    @property
    def SOS_TOKEN_LABEL_ENCODED(self):
        return self._tokenizer.word_index[self.SOS_TOKEN]

    @property
    def EOS_TOKEN_LABEL_ENCODED(self):
        return self._tokenizer.word_index[self.EOS_TOKEN]

    @property
    def vocabs(self):
        word_index = self._tokenizer.word_index
        return sorted(word_index, key=word_index.get)  # Sort by word's index

    @property
    def vocab_size(self):
        return len(self._word_of)

    def fit_on_captions(self, captions_txt):
        captions_txt = self._handle_rare_words(captions_txt)  # discard rare word or maintain
        captions_txt = self._add_eos(captions_txt)   # the output is 'caption zeosz'
        captions_txt = self._add_sos(captions_txt)
        self._tokenizer.fit_on_texts(captions_txt)   # form a dictionary
        self._word_of = {i: w for w, i in self._tokenizer.word_index.items()}  # get the index of each word in the dictionary

    def encode_captions(self, captions_txt):
        captions_txt = self._add_sos(captions_txt)
        captions_txt = self._add_eos(captions_txt)
        return self._tokenizer.texts_to_sequences(captions_txt)

    def decode_captions(self, captions_output, captions_output_expected=None):
        """
        Args
          captions_output: 3-d array returned by a model's prediction; it's the
            same as captions_output returned by preprocess_batch
        """
        captions = captions_output  # Discard the last word (dummy)
        label_encoded = captions.argmax(axis=-1)
        # print(label_encoded)
        num_batches, num_words = label_encoded.shape

        if captions_output_expected is not None:
            caption_lengths = self._caption_lengths(captions_output_expected)
        else:
            caption_lengths = [num_words] * num_batches

        captions_str = []
        for caption_i in range(num_batches):
            caption_str = []
            for word_i in range(caption_lengths[caption_i]):
                label = label_encoded[caption_i, word_i]
                '''remember to change whether to plus one'''
                label += 1  # Real label = label in model + 1
                caption_str.append(self._word_of[label])
            captions_str.append(' '.join(caption_str))
            print(' '.join(caption_str))
        return captions_str

    # TODO Test method below
    def decode_captions_from_list2d(self, captions_encoded):
        """
        Args
          captions_encoded: 1-based (Tokenizer's), NOT 0-based (model's)
        """
        captions_decoded = []
        for caption_encoded in captions_encoded:
            words_decoded = []
            for word_encoded in caption_encoded:
                # No need of incrementing word_encoded
                words_decoded.append(self._word_of[word_encoded])
            captions_decoded.append(' '.join(words_decoded))
        return captions_decoded

    def decode_captions_from_list1d(self, caption_encoded):
        """
        Args
          captions_encoded: 1-based (Tokenizer's), NOT 0-based (model's)
        """
        captions_decoded = []
        words_decoded = []
        for word_encoded in caption_encoded:
            # No need of incrementing word_encoded
            words_decoded.append(self._word_of[word_encoded])
        captions_decoded.append(' '.join(words_decoded))
        return captions_decoded

    def normalize_captions(self, captions_txt):
        captions_txt = self._add_eos(captions_txt)
        return captions_txt

    def preprocess_batch(self, captions_label_encoded):
        captions_input = keras_seq.pad_sequences(captions_label_encoded,
                                           padding='post')
        # Because the number of timesteps/words resulted by the model is
        # maxlen(captions) + 1 (because the first "word" is the image).
        captions_output = [x[1:] for x in captions_input]
        captions_extended1 = keras_seq.pad_sequences(captions_output,
                                                maxlen=captions_input.shape[-1],
                                                padding='post')
        captions_one_hot = list(map(self._tokenizer.sequences_to_matrix,   #
                               np.expand_dims(captions_extended1, -1)))
        captions_one_hot = np.array(captions_one_hot, dtype='int')

        # Decrease/shift word index by 1.
        # Shifting `captions_one_hot` makes the padding word
        # (index=0, encoded=[1, 0, ...]) encoded all zeros ([0, 0, ...]),
        # so its cross entropy loss will be zero.
        captions_decreased = captions_input.copy()
        captions_decreased[captions_decreased > 0] -= 1
        captions_one_hot_shifted = captions_one_hot[:, :, 1:]

        captions_input = captions_decreased
        captions_output = captions_one_hot_shifted
        return captions_input, captions_output

    def _handle_rare_words(self, captions):
        if self._rare_words_handling == 'nothing':
            return captions
        elif self._rare_words_handling == 'discard':
            tokenizer = Tokenizer()
            tokenizer.fit_on_texts(captions)
            new_captions = []
            for caption in captions:
                words = text_to_word_sequence(caption)
                new_words = [w for w in words
                             if tokenizer.word_counts.get(w, 0) >=
                             self._words_min_occur]
                new_captions.append(' '.join(new_words))
            return new_captions
        raise NotImplementedError('rare_words_handling={} is not implemented '
                                  'yet!'.format(self._rare_words_handling))

    def _add_eos(self, captions):
        return list(map(lambda x: x + ' ' + self.EOS_TOKEN, captions))

    def _add_sos(self, captions):
        return list(map(lambda x: self.SOS_TOKEN + ' ' + x, captions))

    def _add_eos_label(self, captions):
        return list(map(lambda x: x + [self.EOS_TOKEN_LABEL_ENCODED], captions))

    def _add_sos_label(self, captions):
        return list(map(lambda x: [self.SOS_TOKEN_LABEL_ENCODED] + x, captions))

    def _caption_lengths(self, captions_output):
        one_hot_sum = captions_output.sum(axis=2)
        return (one_hot_sum != 0).sum(axis=1)