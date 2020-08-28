import os
import sys
import yaml
from datetime import timedelta
import re
class FlickrConfig(object):
    """ Wrapper class for various (hyper)parameters. """
    def __init__(self):
        # about the model architecture
        self.learning_rate = 0.0005
        self.reduce_lr_factor = 0.2
        self.reduce_lr_patience = 3
        self.early_stopping_patience = 10
        self.embedding_dim = 512
        self.hidden_dim = 512
        self.drop_rate = 0.5
        self.rnn_layers = 1
        self.rnn_type = 'lstm'
        self.l1_reg = 0
        self.l2_reg = 0
        self.pretrained_word_vector = None
        self.bidirectional_rnn = False
        # about the hyperparameters
        self.num_epochs = 100
        self.batch_size = 32
        self.val_batch_size = 1
        # self.optimizer = 'Adam'    # 'Adam', 'RMSProp', 'Momentum' or 'SGD'
        self.image_augmentation = False
        self.time_limit = None

        self.trainable_variable = False
        self.rare_words_handling = 'discard'
        self.words_min_occur = 3
        self.max_caption_length = 20
        #for image encoder
        self.img_encoder = 'vgg16'  # 'vgg16'  'vgg19' 'inception_v3' 'resnet50
        self.layer_name = "block5_conv3"  # VGG16 'block5_conv3', VGG19 'block_conv4', Inception_v3, mixed10 resnet50 activation_49
        self.sentence_length = None
        self.img_feature_dim = 512   #vgg16512
        self.img_feature_length = 14*14   #vgg1614*14
        # self.img_encoder = 'resnet101'
        # self.layer_name = "conv5_block3_out"
        # self.sentence_length = None
        # self.img_feature_dim = 2048
        # self.img_feature_length = 7*7

        #about dataset
        self.dataset_root_path = os.path.join('/home/sunjiamei/work/ImageCaptioning/ImgCaptioningExplanation/dataset')

        self.flickr30K_text_dir = os.path.join(self.dataset_root_path, 'flickr30k/Flickr30k_text')
        self.flickr30k_caption_raw_filename = os.path.join(self.flickr30K_text_dir, 'Flickr30k.token')
        self.flickr30k_img_dir = os.path.join(self.dataset_root_path, 'flickr30k/Flickr30k_Dataset')
        self.flickr30k_img_train_filename = os.path.join(self.flickr30K_text_dir, 'Flickr_30k.trainImages.txt')
        self.flickr30k_img_val_filename = os.path.join(self.flickr30K_text_dir, 'Flickr_30k.devImages.txt')
        self.flickr30k_img_test_filename = os.path.join(self.flickr30K_text_dir, 'Flickr_30k.testImages.txt')


    def save_config_as_dict(self, save_path):
        config_dict = dict()
        config_dict['learning_rate'] = self.learning_rate
        config_dict['reduce_lr_factor'] = self.reduce_lr_factor
        config_dict['reduce_lr_patience'] = self.reduce_lr_patience
        config_dict['early_stopping_patience'] = self.early_stopping_patience
        config_dict['embedding_dim'] = self.embedding_dim
        config_dict['hidden_dim'] = self.hidden_dim
        config_dict['drop_rate'] = self.drop_rate
        config_dict['rnn_layers'] = self.rnn_layers
        config_dict['rnn_type'] = self.rnn_type
        config_dict['l1_reg'] = self.l1_reg
        config_dict['l2_reg'] = self.l2_reg
        config_dict['pretrained_word_vector'] = self.pretrained_word_vector
        config_dict['biderectional_rnn'] = self.bidirectional_rnn
        config_dict['num_epochs'] = self.num_epochs
        config_dict['batch_size'] = self.batch_size
        config_dict['time_limit'] = self.time_limit
        config_dict['rare_words_handling'] = self.rare_words_handling
        config_dict['words_min_occur'] = self.words_min_occur
        config_dict['img_encoder'] = self.img_encoder
        config_dict['layer_name'] = self.layer_name
        config_dict['img_feature_dim'] = self.img_feature_dim
        config_dict['img_feature_length'] = self.img_feature_length
        time_limit = config_dict['time_limit']
        if time_limit:
            config_dict['time_limit'] = str(time_limit)

        file_name = 'config.yaml'
        with open(os.path.join(save_path,file_name), 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

class COCOConfig(object):
    
    def __init__(self):
        self.learning_rate = 0.0005 #0.0005050182149478269
        self.reduce_lr_factor = 0.5
        self.reduce_lr_patience = 3
        self.early_stopping_patience = 6
        self.embedding_dim = 512
        self.hidden_dim = 512
        self.drop_rate = 0.5
        self.rnn_layers = 1
        self.rnn_type = 'lstm'
        self.l1_reg = 0
        self.l2_reg = 0
        self.pretrained_word_vector = None #'glove'
        self.bidirectional_rnn = False
        self.num_epochs = 100
        self.batch_size = 20
        self.image_augmentation = False
        self.time_limit = None

        self.trainable_variable = False
        self.rare_words_handling = 'discard'
        self.words_min_occur = 5

        self.max_caption_length = 20
        #for image encoder
        # inception_v3
        # self.img_encoder = 'inception_v3'  # 'vgg16'  'vgg19' 'inception_v3'
        # self.layer_name = "mixed10"  # VGG16 'block5_conv3', VGG19 'block_conv4', Inception_v3, mixed10
        # self.sentence_length = None
        # self.img_feature_dim = 2048
        # self.img_feature_length = 5*5
        # resnet50
        # self.img_encoder = 'resnet50'  # 'vgg16'  'vgg19' 'inception_v3' 'resnet50
        # self.layer_name = "activation_98"  # VGG16 'block5_conv3', VGG19 'block_conv4', Inception_v3, mixed10 resnet50 activation_98
        # self.sentence_length = None
        # self.img_feature_dim = 2048#vgg16512
        # self.img_feature_length = 7*7#vgg1614*14
        # VGG 16
        self.img_encoder = 'vgg16'  # 'vgg16'  'vgg19' 'inception_v3' 'resnet50
        self.layer_name = "block5_conv3"  # VGG16 'block5_conv3', VGG19 'block_conv4', Inception_v3, mixed10 resnet50 activation_49
        self.sentence_length = None
        self.img_feature_dim = 512   #vgg16512
        self.img_feature_length = 14*14   #vgg1614*14
        # resnet101
        # self.img_encoder = 'resnet101'
        # self.layer_name = "conv5_block3_out"
        # self.sentence_length = None
        # self.img_feature_dim = 2048
        # self.img_feature_length = 7*7


        #about dataset
        self.dataset_root_path = os.path.join('/home/sunjiamei/work/ImageCaptioning/ImgCaptioningExplanation/dataset')
        self.coco_caption_dir = os.path.join(self.dataset_root_path, 'coco/annotations')
        self.coco_caption_train_file = os.path.join(self.coco_caption_dir, 'captions_train2017.json')
        self.coco_caption_val_file = os.path.join(self.coco_caption_dir, 'captions_train2017.json')
        self.coco_caption_test_file = os.path.join(self.coco_caption_dir, 'captions_val2017.json')
        self.coco_img_dir = os.path.join(self.dataset_root_path, 'coco/images')
        self.coco_img_train_dir = os.path.join(self.coco_img_dir, 'train2017')
        self.coco_img_val_dir = os.path.join(self.coco_img_dir, 'train2017')
        self.coco_img_test_dir = os.path.join(self.coco_img_dir, 'val2017')
        self.coco_train_length = 110000
        self.coco_val_length = 5000
        self.coco_test_lengt = 5000

    def save_config_as_dict(self, save_path):
        config_dict = dict()
        config_dict['learning_rate'] = self.learning_rate
        config_dict['reduce_lr_factor'] = self.reduce_lr_factor
        config_dict['reduce_lr_patience'] = self.reduce_lr_patience
        config_dict['early_stopping_patience'] = self.early_stopping_patience
        config_dict['embedding_dim'] = self.embedding_dim
        config_dict['hidden_dim'] = self.hidden_dim
        config_dict['drop_rate'] = self.drop_rate
        config_dict['rnn_layers'] = self.rnn_layers
        config_dict['rnn_type'] = self.rnn_type
        config_dict['l1_reg'] = self.l1_reg
        config_dict['l2_reg'] = self.l2_reg
        config_dict['pretrained_word_vector'] = self.pretrained_word_vector
        config_dict['biderectional_rnn'] = self.bidirectional_rnn
        config_dict['num_epochs'] = self.num_epochs
        config_dict['batch_size'] = self.batch_size
        config_dict['time_limit'] = self.time_limit
        config_dict['rare_words_handling'] = self.rare_words_handling
        config_dict['words_min_occur'] = self.words_min_occur
        config_dict['img_encoder'] = self.img_encoder
        config_dict['layer_name'] = self.layer_name
        config_dict['img_feature_dim'] = self.img_feature_dim
        config_dict['img_feature_length'] = self.img_feature_length
        time_limit = config_dict['time_limit']
        if time_limit:
            config_dict['time_limit'] = str(time_limit)

        file_name = 'config.yaml'
        with open(os.path.join(save_path,file_name), 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

class COCO2014Config(COCOConfig):
    def __init__(self):
        super(COCO2014Config, self).__init__()
        self.dataset_root_path = os.path.join(os.path.dirname(__file__), 'dataset')
        self.dataset_file = os.path.join(self.dataset_root_path, 'coco2014/dataset_coco.json')
        self.coco2014_img_dir = os.path.join(self.dataset_root_path, 'coco/images')
        self.batch_size = 20


