import json
import os
import re
from models import io_utils
import config
from collections import defaultdict, namedtuple
from itertools import *
from math import ceil
from copy import copy
from operator import attrgetter
from models.preprocessors import  ImagePreprocessor, CaptionPreprocessorAttention
import xml.etree.ElementTree as ET

'''Load the dataset as the format of Data_format including training set, val set and test set'''
Data_format = namedtuple('Datum', 'img_filename img_path caption_txt all_captions_txt')


class COCOCategory(object):
    def __init__(self, file_path):
        if os.path.isfile(file_path):
            with open(file_path, 'r') as f:
                dataset = json.load(f)      # ['images', 'licenses', 'annotations', 'info', 'categories']
                f.close()
            self.images = dataset['images']      # dict_keys(['license', 'coco_url', 'width', 'flickr_url', 'file_name', 'height', 'date_captured', 'id'])
            self.annotations = dataset['annotations']     # dict_keys(['iscrowd', 'area', 'image_id', 'segmentation', 'bbox', 'category_id', 'id'])
            self.categories = dataset['categories']  #['name', 'id', 'supercategory']
        else:
            raise NotImplementedError('The input path is invalid')
        self.filename_to_category = dict()
        self._build_category()

    def _build_category(self):
        '''dict:  key - img_name, '''
        id_to_file_name = dict()
        for img_item in self.images:
            if img_item['id'] not in id_to_file_name.keys():
                id_to_file_name[img_item['id']] = dict()
            id_to_file_name[img_item['id']]['file_name'] = img_item['file_name']
            id_to_file_name[img_item['id']]['shape'] = (img_item['width'], img_item['height'])
        categoryid_to_categories = dict()
        for category_item in self.categories:
            categoryid_to_categories[category_item['id']] = category_item['name']
        imgid_to_categorynames = dict()
        imgid_to_bbbox = dict()
        for annotation_item in self.annotations:
            imgid = annotation_item['image_id']
            cate_id = annotation_item['category_id']
            if imgid not in imgid_to_categorynames.keys():
                imgid_to_categorynames[annotation_item['image_id']] = dict()
            imgid_to_categorynames[annotation_item['image_id']][categoryid_to_categories[cate_id]]=str(cate_id)
            if imgid not in imgid_to_bbbox.keys():
                imgid_to_bbbox[imgid] = dict()
            if cate_id not in imgid_to_bbbox[imgid].keys():
                imgid_to_bbbox[imgid][cate_id] = []
            coordinates= annotation_item['bbox']
            xmin = coordinates[0]
            ymin = coordinates[1]
            xmax = coordinates[2] + xmin
            ymax = coordinates[3] + ymin
            new_bbox = [xmin, ymin, xmax, ymax]
            imgid_to_bbbox[imgid][cate_id].append(new_bbox)  # bbox is [x, y, xmax, ymax] x is for width dimension
        for imgid in imgid_to_categorynames.keys():
            filename = id_to_file_name[imgid]['file_name']
            shape = id_to_file_name[imgid]['shape']
            ratio = (224/shape[0], 224/shape[1])
            self.filename_to_category[filename] = dict()
            self.filename_to_category[filename]['categories'] = imgid_to_categorynames[imgid]
            self.filename_to_category[filename]['bbox'] = imgid_to_bbbox[imgid]
            self.filename_to_category[filename]['shape'] = shape
            self.filename_to_category[filename]['resize_ratio'] = ratio
        with open('./dataset/coco/COCOvalEntities.json', 'w') as f:
            json.dump(self.filename_to_category, f)
            f.close()
        return self.filename_to_category


class Flickr30kCategory(object):
    def __init__(self, root_path):
        self._root_path = root_path
        self._sentence_root_path = root_path + 'Sentences/'
        self._annotations_root_path = root_path + 'Annotations/'
        sentences_list = self._sentence_root_path + 'list.txt'
        with open(sentences_list, 'r') as f:
            self._sentence_files = f.readlines()
        self.filename_to_category = dict()

    def _build(self):
        for fn in self._sentence_files:
            filename = fn.strip('\n').split('.')[0]
            img_filename = filename + '.jpg'
            self.filename_to_category[img_filename] = dict()
            img_bbox = dict()
            sentence_datas = self.get_sentence_data(self._sentence_root_path + filename + '.txt')  # a list of sentence_data
            annotation = self.get_annotations(self._annotations_root_path + filename + '.xml')  #['scene', 'height', 'nobox', 'boxes', 'depth', 'width']
            self.filename_to_category[img_filename]['shape'] = (annotation['width'], annotation['height'])
            self.filename_to_category[img_filename]['resize_ratio'] = (224.0 / annotation['width'], 224.0 / annotation['height'])
            pharse_id_to_phrases = dict()
            phrase_to_phrase_id = dict()
            boxes_id = annotation['boxes'].keys()
            for box_id in boxes_id:
                img_bbox[box_id] = annotation['boxes'][box_id]
            for sentence_data in sentence_datas:
                phrases = sentence_data['phrases']
                for phrase in phrases:
                    id = phrase['phrase_id']
                    if id not in img_bbox.keys():
                        continue
                    if id not in pharse_id_to_phrases.keys():
                        pharse_id_to_phrases[id] = []
                    pharse_id_to_phrases[id].append(phrase['phrase'])
                    if phrase['phrase'] not in phrase_to_phrase_id.keys():
                        phrase_to_phrase_id[phrase['phrase']] = id
            self.filename_to_category[img_filename]['categories'] = phrase_to_phrase_id
            self.filename_to_category[img_filename]['bbox'] = img_bbox
        with open(self._root_path + 'Flickr30kEntities.json', 'w') as f:
            json.dump(self.filename_to_category, f)
            f.close()

        with open(self._root_path + 'Flickr30kEntities.json', 'r') as f:
            filedict = json.load(f)
            f.close()
        for key in filedict.keys():
            for k in filedict[key].keys():
                print(k)
                print(filedict[key][k])
            break

    def get_sentence_data(self, fn):
        """
        Parses a sentence file from the Flickr30K Entities dataset

        input:
          fn - full file path to the sentence file to parse

        output:
          a list of dictionaries for each sentence with the following fields:
              sentence - the original sentence
              phrases - a list of dictionaries for each phrase with the
                        following fields:
                          phrase - the text of the annotated phrase
                          first_word_index - the position of the first word of
                                             the phrase in the sentence
                          phrase_id - an identifier for this phrase
                          phrase_type - a list of the coarse categories this
                                        phrase belongs to

        """
        with open(fn, 'r') as f:
            sentences = f.read().split('\n')
        annotations = []
        for sentence in sentences:
            if not sentence:
                continue
            first_word = []
            phrases = []
            phrase_id = []
            phrase_type = []
            words = []
            current_phrase = []
            add_to_phrase = False
            for token in sentence.split():
                if add_to_phrase:
                    if token[-1] == ']':
                        add_to_phrase = False
                        token = token[:-1]
                        current_phrase.append(token)
                        phrases.append(' '.join(current_phrase))
                        current_phrase = []
                    else:
                        current_phrase.append(token)

                    words.append(token)
                else:
                    if token[0] == '[':
                        add_to_phrase = True
                        first_word.append(len(words))
                        parts = token.split('/')
                        phrase_id.append(parts[1][3:])
                        phrase_type.append(parts[2:])
                    else:
                        words.append(token)

            sentence_data = {'sentence': ' '.join(words), 'phrases': []}
            for index, phrase, p_id, p_type in zip(first_word, phrases, phrase_id, phrase_type):
                sentence_data['phrases'].append({'first_word_index': index,
                                                 'phrase': phrase,
                                                 'phrase_id': p_id,
                                                 'phrase_type': p_type})
            annotations.append(sentence_data)
        return annotations

    def get_annotations(self, fn):
        """
        Parses the xml files in the Flickr30K Entities dataset

        input:
          fn - full file path to the annotations file to parse

        output:
          dictionary with the following fields:
              scene - list of identifiers which were annotated as
                      pertaining to the whole scene
              nobox - list of identifiers which were annotated as
                      not being visible in the image
              boxes - a dictionary where the fields are identifiers
                      and the values are its list of boxes in the
                      [xmin ymin xmax ymax] format
        """
        tree = ET.parse(fn)
        root = tree.getroot()
        size_container = root.findall('size')[0]
        anno_info = {'boxes': {}, 'scene': [], 'nobox': []}
        for size_element in size_container:
            anno_info[size_element.tag] = int(size_element.text)
        for object_container in root.findall('object'):
            for names in object_container.findall('name'):
                box_id = names.text
                box_container = object_container.findall('bndbox')
                if len(box_container) > 0:
                    if box_id not in anno_info['boxes']:
                        anno_info['boxes'][box_id] = []
                    xmin = int(box_container[0].findall('xmin')[0].text) - 1
                    ymin = int(box_container[0].findall('ymin')[0].text) - 1
                    xmax = int(box_container[0].findall('xmax')[0].text) - 1
                    ymax = int(box_container[0].findall('ymax')[0].text) - 1
                    anno_info['boxes'][box_id].append([xmin, ymin, xmax, ymax])
                else:
                    nobndbox = int(object_container.findall('nobndbox')[0].text)
                    if nobndbox > 0:
                        anno_info['nobox'].append(box_id)

                    scene = int(object_container.findall('scene')[0].text)
                    if scene > 0:
                        anno_info['scene'].append(box_id)
        return anno_info


class Dataset(object):
    _DATASET_DIR_NAME = '../dataset'
    _TRAINING_RESULTS_DIR_NAME = 'training-results'

    def __init__(self, dataset_name, lemmatize_caption=False, single_caption=False):
        self.dataset_name = dataset_name
        self._lemmatize_caption = lemmatize_caption
        self._single_caption = single_caption
        self._root_result_path = io_utils.path_from_results_dir(dataset_name)
        self._create_dirs()

    @property
    def training_set(self):
        return self._training_set

    @property
    def validation_set(self):
        return self._validation_set

    @property
    def test_set(self):
        return self._test_set

    @property
    def training_set_size(self):
        return len(self._training_set)

    @property
    def validation_set_size(self):
        return len(self._validation_set)

    @property
    def test_set_size(self):
        return len(self._test_set)

    @property
    def dataset_dir(self):
        return os.path.join(io_utils._DATASET_ROOT_DIR, self.dataset_name)

    @property
    def training_results_dir(self):
        return os.path.join(self._root_result_path, self._TRAINING_RESULTS_DIR_NAME)

    def _create_dirs(self):
        io_utils.mkdir_p(self.training_results_dir)


class COCODataset(Dataset):
    DATASET_NAME = 'coco'

    def __init__(self, single_caption=False):
        super(COCODataset, self).__init__(self.DATASET_NAME, single_caption=single_caption)
        self.config = config.COCOConfig()
        self.caption_train_file = self.config.coco_caption_train_file
        self.caption_val_file = self.config.coco_caption_val_file
        self.caption_test_file = self.config.coco_caption_test_file
        self.img_train_dir = self.config.coco_img_train_dir
        self.img_val_dir = self.config.coco_img_val_dir
        self.img_test_dir = self.config.coco_img_test_dir
        self.single_caption = single_caption

        self._training_set = self._build(self.caption_train_file, self.img_train_dir, mode='train')
        print('COCO Traing set is built', len(self._training_set))
        self._validation_set = self._build(self.caption_val_file, self.img_val_dir, mode='val')
        print('COCO Validation set is built', len(self._validation_set))
        self._test_set = self._build(self.caption_test_file, self.img_test_dir, mode='test')
        print('COCO Test set is built', len(self._test_set))

    def _build(self, annotation_file=None, img_dir=None, mode='train'):
        # :param annotation_file: the json file that contains the images info and annotations info
        # :param img_dir: the directory to the image files
        # :param mode: mode is a string, 'train', 'val' and 'test'.  We split 100000 and 8000 imgs in coco_train2017 for
        #              training and validation and use the coco val2017 dataset as the test set
        # :return: return a list of namedtuple including the image filename, img path and its corresponding captions
        #          the captions are raw text.
        coco_anns = json.load(open(annotation_file, 'r'))  # load the raw json coco annotations file)
        coco_anns = self._process_dataset(coco_anns)       # add '.' to the end of each captions
        imgID2captions = defaultdict(list)                 # form a dict with imgID as keys and caption list as values
        imgID2imgFilename_path = defaultdict(dict)         # form a dict with imgID as keys and a dict as values, the
        # img_ids = []                                     # dict contains img file name and the path.
        for ann in coco_anns['annotations']:
            imgID2captions[ann['image_id']].append(ann['caption'])
        for img in coco_anns['images']:
            imgID2imgFilename_path[img['id']]['file_name'] = img['file_name']
            imgID2imgFilename_path[img['id']]['img_path'] = os.path.join(img_dir, img['file_name'])
        imgID2imgFilename_path = dict(imgID2imgFilename_path)
        imgID2captions = dict(imgID2captions)
        dataset = []    # a list of Data_format
        img_ids = list(imgID2captions.keys())
        img_ids.sort()
        if mode == 'train':
            img_ids = img_ids[:self.config.coco_train_length]
        if mode == 'val':
            img_ids = img_ids[self.config.coco_train_length:self.config.coco_train_length + self.config.coco_val_length]
        if mode == 'test':
            assert len(img_ids) == self.config.coco_test_lengt
        for img_id in img_ids:
            img_filename = imgID2imgFilename_path[img_id]['file_name']
            img_path = imgID2imgFilename_path[img_id]['img_path']
            all_captions = imgID2captions[img_id]
            for caption in all_captions:
                dataset.append(Data_format(img_filename=img_filename,
                                           img_path=img_path,
                                           caption_txt=caption,
                                           all_captions_txt=all_captions))
                if self.single_caption:
                    break
        return dataset

    def _process_dataset(self, ann_file):
        for ann in ann_file['annotations']:
            q = ann['caption'].lower()
            if q[-1]!='.':
                q = q + '.'
            ann['caption'] = q
        return ann_file


class COCODatasetKarpathy(Dataset):
    DATASET_NAME = 'coco2014'

    def __init__(self, single_caption=False):
        super(COCODatasetKarpathy, self).__init__(self.DATASET_NAME, single_caption=single_caption)
        self.config = config.COCO2014Config()
        self.dataset_file = self.config.dataset_file
        self.img_dir = self.config.coco2014_img_dir
        self.single_caption = single_caption

        self.dataset_raw = json.load(open(self.dataset_file, 'r'))

        self._training_set, self._test_set, self._validation_set = self._build(self.dataset_raw)
        print('COCO2014 Traing set is built')
        print(len(self._training_set))
        print('COCO2014 Validation set is built')
        print(len(self._validation_set))
        print('COCO2014 Test set is built')
        print(len(self._test_set))


    def _build(self, dataset_raw):
        print(self.DATASET_NAME)
        trainset = []
        testset = []
        valset = []
        for item in dataset_raw['images']:
            split = item['split']
            filename = item['filename']
            file_path = item['filepath']
            img_id = item['cocoid']
            # print(img_id)
            img_path = os.path.join(self.img_dir, file_path, filename)
            all_captions = []
            for sentence in item['sentences']:
                words = sentence['tokens']
                all_captions.append(' '.join(words))
            if split in ['train', 'restval']:
                for caption in all_captions:
                    trainset.append(Data_format(img_filename=img_id,
                                               img_path=img_path,
                                               caption_txt=caption,
                                               all_captions_txt=all_captions))
                    if self.single_caption:
                        break
            elif split in ['val']:
                for caption in all_captions:
                    valset.append(Data_format(img_filename=img_id,
                                               img_path=img_path,
                                               caption_txt=caption,
                                               all_captions_txt=all_captions))
                    # if self.single_caption:
                    break
            else:
                for caption in all_captions:
                    testset.append(Data_format(img_filename=img_id,
                                               img_path=img_path,
                                               caption_txt=caption,
                                               all_captions_txt=all_captions))
                    # if self.single_caption:
                    break
        return trainset, testset, valset


class Flickr30kDataset(Dataset):

    DATASET_NAME = 'flickr30k'
    def __init__(self, config, lemmatize_caption=False, single_caption=False):
        super(Flickr30kDataset, self).__init__(self.DATASET_NAME, lemmatize_caption=lemmatize_caption, single_caption=single_caption)
        self.config = config
        self.caption_raw_filename = self.config.flickr30k_caption_raw_filename
        self.img_train_filename = self.config.flickr30k_img_train_filename
        self.img_val_filename = self.config.flickr30k_img_val_filename
        self.img_test_filename = self.config.flickr30k_img_test_filename
        self._build()

    def _build(self):
        print(self.DATASET_NAME)
        self._captions_of = self._build_captions()
        self._training_set = self._build_set(self.img_train_filename)
        print('Flickr Traing set is built', len(self._training_set))
        self._validation_set = self._build_set(self.img_val_filename)
        print('Flickr Val set is built', len(self._validation_set))
        self._test_set = self._build_set(self.img_test_filename)
        print('Flickr Test set is built', len(self._test_set))

    def _build_captions(self):

        caption_filename = self.caption_raw_filename
        lines = io_utils.read_text_file(caption_filename)
        lines_splitted = map(lambda x: re.split(r'#\d\t', x), lines)  #'#\d\t' is the splition between img_filename and the caption
        lines_splitted = list(lines_splitted)
        captions_of = defaultdict(list)
        for img_filename, caption_txt in lines_splitted:
            caption_txt = self._process_dataset(caption_txt)
            captions_of[img_filename].append(caption_txt)
        return dict(captions_of)
    def _build_set(self, img_set_filename):
        img_filenames = io_utils.read_text_file(img_set_filename)
        dataset = []
        for img_filename in img_filenames:
            img_path = os.path.join(self.config.flickr30k_img_dir, img_filename)
            if img_filename not in self._captions_of.keys():
                continue
            all_captions_txt = self._captions_of[img_filename]
            for caption_txt in all_captions_txt:
                dataset.append(Data_format(img_filename=img_filename,
                                           img_path=img_path,
                                           caption_txt=caption_txt,
                                           all_captions_txt=all_captions_txt))
                if self._single_caption:
                    break
        return dataset

    def _process_dataset(self, caption_txt):
        q = caption_txt.lower()
        if q[-1]!='.':
            q = q + '.'
        caption_txt = q
        return caption_txt


'''encode the images and embed the captions, generate the pair of training input X and output Y '''
class DatasetPreprocessorAttention(object):
    def __init__(self, dataset, config, single_caption=False):
        self.config = config
        self.batch_size =config.batch_size
        self.dataset = dataset
        self.single_caption = single_caption
        self.image_preprocessor = ImagePreprocessor(config.img_encoder, config.image_augmentation)
        self.caption_preprocessor = CaptionPreprocessorAttention(self.dataset.dataset_name, config)
        self.build_vocabulary()

    @property
    def vocabs(self):
        return self.caption_preprocessor.vocabs

    @property
    def vocab_size(self):
        return self.caption_preprocessor.vocab_size

    @property
    def training_steps(self):
        return int(ceil(1. * self.dataset.training_set_size /
                        self.batch_size))

    @property
    def validation_steps(self):
        return int(ceil(1. * self.dataset.validation_set_size /
                        self.batch_size))

    @property
    def test_steps(self):
        return int(ceil(1. * self.dataset.test_set_size /
                        self.batch_size))

    @property
    def training_results_dir(self):
        return self.dataset.training_results_dir

    def training_set(self, include_datum=False):
        for batch in self._batch_generator(self.dataset.training_set,
                                           include_datum,
                                           random_transform=True):
            yield batch

    def validation_set(self, include_datum=False):
        for batch in self._batch_generator(self.dataset.validation_set,
                                           include_datum,
                                           random_transform=False):
            yield batch

    def test_set(self, include_datum=False):
        for batch in self._batch_generator(self.dataset.test_set,
                                           include_datum,
                                           random_transform=False):
            yield batch

    def build_vocabulary(self):
        training_set = self.dataset.training_set# + self.dataset.test_set + self.dataset.validation_set
        if self.single_caption:
            training_captions = map(attrgetter('all_captions_txt'),
                                    training_set)
            training_captions = list(training_captions)
            training_captions = list(chain.from_iterable(training_captions))
        else:
            training_captions = map(attrgetter('caption_txt'), training_set)
            training_captions = list(training_captions)
        self.caption_preprocessor.fit_on_captions(training_captions)
        print('vocabulary size is: ', self.caption_preprocessor.vocab_size)
    def _batch_generator(self, datum_list, include_datum=False,
                         random_transform=True):
        # TODO Make it thread-safe. Currently only suitable for workers=1 in

        datum_list = copy(datum_list)
        while True:
            datum_batch = []
            for datum in datum_list:
                datum_batch.append(datum)
                if len(datum_batch) >= self.batch_size:
                    yield self._preprocess_batch(datum_batch, include_datum,
                                                 random_transform)
                    datum_batch = []
            if datum_batch:
                yield self._preprocess_batch(datum_batch, include_datum,
                                             random_transform)               # the last batch may not fullfill the batch_size


    def _preprocess_batch(self, datum_batch, include_datum=False,
                          random_transform=True):

        imgs_path = map(attrgetter('img_path'), datum_batch)
        imgs_path = list(imgs_path)
        captions_txt = map(attrgetter('caption_txt'), datum_batch)
        captions_txt = list(captions_txt)
        img_batch = self.image_preprocessor.preprocess_images(imgs_path,
                                                            random_transform)
        caption_batch = self.caption_preprocessor.encode_captions(captions_txt)  # split the text to word list with label

        imgs_input = self.image_preprocessor.preprocess_batch(img_batch)
        captions = self.caption_preprocessor.preprocess_batch(caption_batch)

        captions_input, captions_output = captions
        X, y = [captions_input, imgs_input], captions_output
        if include_datum:
            return X, y, datum_batch
        else:
            return X, y




