import heapq
import os
from collections import namedtuple
import glob
from time import sleep
from tqdm import tqdm
from keras.utils import GeneratorEnqueuer
from preparedataset import Flickr30kDataset, COCODataset, DatasetPreprocessorAttention

from io_utils import logging, write_yaml_file
from metrics import BLEU, CIDEr, METEOR, ROUGE, SPICE, BERT
from itertools import zip_longest
from model import *
import config
import pickle

MAXIMUM_LENGTH = 20


class BasicInference(object):
    """A very basic inference without beam search. Technically, it is not an
    inference because the actual captions are also feeded into the model."""


    _MAX_Q_SIZE = 10
    _WORKERS = 1
    _WAIT_TIME = 0.01

    def __init__(self, keras_model, dataset_provider, test_dataset_provider):
        self._model = keras_model
        self._dataset_provider = test_dataset_provider
        self._preprocessor = dataset_provider.caption_preprocessor
        self._metrics = [BLEU(4), METEOR(),  CIDEr(), ROUGE(),SPICE(), BERT()] #, ROUGE(),SPICE(), BERT()
        self._max_caption_length = 20
        self._beam_size = 1

    def predict_training_set(self, include_datum=True):
        return self._predict(self._dataset_provider.training_set,
                             self._dataset_provider.training_steps,
                             include_datum)

    def predict_validation_set(self, include_datum=True):
        return self._predict(self._dataset_provider.validation_set,
                             self._dataset_provider.validation_steps,
                             include_datum)

    def predict_test_set(self, include_datum=True):
        return self._predict(self._dataset_provider.test_set,
                             # 1,
                             self._dataset_provider.test_steps,
                             include_datum)

    def evaluate_training_set(self, include_prediction=False):
        return self._evaluate(self.predict_training_set(include_datum=True),
                              include_prediction=include_prediction)

    def evaluate_validation_set(self, include_prediction=False):
        return self._evaluate(self.predict_validation_set(include_datum=True),
                              include_prediction=include_prediction)

    def evaluate_test_set(self, include_prediction=False):
        return self._evaluate(self.predict_test_set(include_datum=True),
                              include_prediction=include_prediction)

    def _predict(self,
                 data_generator_function,
                 steps_per_epoch,
                 include_datum=True):
        data_generator = data_generator_function(include_datum=True)
        enqueuer = GeneratorEnqueuer(data_generator)
        enqueuer.start(workers=self._WORKERS, max_queue_size=self._MAX_Q_SIZE)

        caption_results = []
        datum_results = []
        for _ in tqdm(range(steps_per_epoch)):
            generator_output = None
            while enqueuer.is_running():
                if not enqueuer.queue.empty():
                    generator_output = enqueuer.get()
                    break
                else:
                    sleep(self._WAIT_TIME)

            X, y, datum_batch = next(generator_output)
            captions_pred_str = self._predict_batch(X, y)
            caption_results += captions_pred_str
            datum_results += datum_batch

        enqueuer.stop()

        if include_datum:
            return zip(caption_results, datum_results)
        else:
            return caption_results

    def _predict_batch(self, X, y):
        _, imgs_input = X
        batch_size = imgs_input.shape[0]
        EOS_ENCODED = self._dataset_provider.caption_preprocessor.EOS_TOKEN_LABEL_ENCODED
        SOS_ENCODED = self._dataset_provider.caption_preprocessor.SOS_TOKEN_LABEL_ENCODED
        cap = [[SOS_ENCODED] for _ in range(batch_size)]
        for _ in range(self._max_caption_length):
            captions_input, _ = self._dataset_provider.caption_preprocessor.preprocess_batch(cap)
            preds = self._model.predict_on_batch([captions_input, imgs_input])
            preds = self._log_softmax(preds)
            preds = preds[:, -1]  # We only care the last word in a caption
            top_words = np.argmax(preds, axis=-1)
            top_words = top_words
            for i in range(batch_size):
                if EOS_ENCODED in cap[i]:
                    continue
                cap[i] = cap[i]+ [top_words[i]]
        results = []
        for i in range(batch_size):
            cap_str = []
            for j in range(1,len(cap[i])-1):
                word = self._dataset_provider.caption_preprocessor._word_of[cap[i][j]]
                cap_str.append(word)
            results.append(' '.join(cap_str))
        return results

    def _log_softmax(self, x):
        x = x - np.max(x, axis=-1, keepdims=True)  # For numerical stability
        return x - np.log(np.sum(np.exp(x), axis=-1, keepdims=True))

    def _evaluate(self, caption_datum_pairs, include_prediction=False):
        id_to_prediction = {}
        id_to_references = {}
        id_to_prediction_coco = []
        processed_id = []
        for caption_pred, datum in caption_datum_pairs:
            img_id = datum.img_filename
            if img_id in processed_id:
                continue
            else:
                processed_id.append(img_id)
            caption_expected = self._preprocessor.normalize_captions(
                                                        datum.all_captions_txt)
            id_to_references[img_id] = []
            for cap in caption_expected:
                id_to_references[img_id].append({'caption':cap})
            id_to_prediction[img_id] = [{'caption':caption_pred}]
            id_to_prediction_coco.append({'image_id':img_id, 'caption': caption_pred})
        metrics = {}
        for metric in self._metrics:
            metric_name_to_value = metric.calculate(id_to_prediction,
                                                    id_to_references)
            metrics.update(metric_name_to_value)
        print(metrics)
        return (metrics, id_to_prediction, id_to_references) if include_prediction else metrics


class BeamSearchInference(BasicInference):
    """An implementation of inference using beam search."""
    def __init__(self,
                 keras_model,
                 dataset,
                 test_dataset,
                 config,
                 test_config,
                 beam_size=3,
                 max_caption_length=20):
        print(dataset.DATASET_NAME)
        dataset_provider = DatasetPreprocessorAttention(dataset, config, single_caption=True)  # change the DatasetProcessor according to the model
        test_dataset_provider = DatasetPreprocessorAttention(test_dataset, test_config, single_caption=True)
        super(BeamSearchInference, self).__init__(keras_model,
                                                  dataset_provider, test_dataset_provider)
        self._beam_size = beam_size
        # print(beam_size)
        self._max_caption_length = max_caption_length

    def _predict_batch(self, X, y):
        _, imgs_input = X

        batch_size = imgs_input.shape[0]

        EOS_ENCODED = self._preprocessor.EOS_TOKEN_LABEL_ENCODED
        SOS_ENCODED = self._preprocessor.SOS_TOKEN_LABEL_ENCODED

        complete_captions = BatchNLargest(batch_size=batch_size,
                                          n=self._beam_size)
        partial_captions = BatchNLargest(batch_size=batch_size,
                                         n=self._beam_size)
        partial_captions.add([Caption(sentence_encoded=[SOS_ENCODED, EOS_ENCODED],
                                      log_prob=0.0)
                              for _ in range(batch_size)])

        for _ in range(self._max_caption_length):
            partial_captions_prev = partial_captions
            partial_captions = BatchNLargest(batch_size=batch_size,
                                             n=self._beam_size)

            for top_captions in partial_captions_prev.n_largest():
                sentences_encoded = [x.sentence_encoded for x in top_captions]
                captions_input, _= self._preprocessor.preprocess_batch(sentences_encoded)
                preds = self._model.predict_on_batch([captions_input, imgs_input])
                preds = preds[:, :-1, :]  # Discard the last word (dummy) comment this line if use attention model. attention model doesn't add the dummy word
                preds = preds[:, -1]  # We only care the last word in a caption
                preds = self._log_softmax(preds)
                top_words = np.argpartition(preds, -self._beam_size)[:, -self._beam_size:]
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
                        sentence_encoded = sentence[:-1] + [word, EOS_ENCODED]
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
        if len(complete_captions.n_largest()) > 0:
            top_complete_captions = complete_captions.n_largest(sort=True)[0]

            results = []
            for partial_caption, complete_caption in zip(top_partial_captions,
                                                         top_complete_captions):
                if complete_caption is None:
                    results.append(partial_caption.sentence_encoded[1:])
                else:
                    results.append(complete_caption.sentence_encoded[1:])
        else:
            results = []
            for partial_caption in top_partial_captions:
                results.append(partial_caption.sentence_encoded[1:])

        print(self._preprocessor.decode_captions_from_list2d(results))
        return self._preprocessor.decode_captions_from_list2d(results)

    def _log_softmax(self, x):
        x = x - np.max(x, axis=-1, keepdims=True)  # For numerical stability
        return x - np.log(np.sum(np.exp(x), axis=-1, keepdims=True))

    def softmax(self,x):
        """Compute the softmax in a numerically stable way."""
        x = x - np.max(x)
        exp_x = np.exp(x)
        softmax_x = exp_x / np.sum(exp_x)
        return softmax_x


class BatchNLargest(object):
    """A batch priority queue."""

    def __init__(self, batch_size, n):
        self._batch_size = batch_size
        self._n_largests = [NLargest(n=n) for _ in range(batch_size)]

    def add(self, items):
        if len(items) != self._batch_size:
            raise ValueError('len of items must be equal to batch_size!')
        for n_largest, item in zip(self._n_largests, items):
            n_largest.add(item)

    def add_many(self, itemss):
        if len(itemss) != self._batch_size:
            raise ValueError('len of itemss must be equal to batch_size!')
        for n_largest, items in zip(self._n_largests, itemss):
            n_largest.add_many(items)

    def n_largest(self, sort=True):
        result = [x.n_largest(sort=sort) for x in self._n_largests]
        result_transpose = zip_longest(*result, fillvalue=None)
        return list(result_transpose) # Transpose result


class NLargest(object):
    """An implementation of priority queue with max size."""

    def __init__(self, n):
        self._n = n
        self._heap = []

    def add(self, item):
        if item is None:
            return
        if len(self._heap) < self._n:
            heapq.heappush(self._heap, item)
        else:
            heapq.heappushpop(self._heap, item)

    def add_many(self, items):
        for item in items:
            self.add(item)

    def n_largest(self, sort=True):
        return sorted(self._heap, reverse=True) if sort else self._heap


Caption = namedtuple('Caption', 'log_prob sentence_encoded')


def main(training_dir, dataset, test_dataset, config, test_config, model_name, model_type='gridTD',
         dataset_type='validation',
         beam_size=3,
         max_caption_length=20):
    print(dataset.DATASET_NAME)
    # if method != 'beam_search':
    #     raise NotImplementedError('inference method = {} is not implemented '
    #                               'yet!'.format(method))
    if dataset_type not in ['validation', 'test']:
        raise ValueError('dataset_type={} is not recognized!'.format(
                                                                dataset_type))
    model_path = os.path.join(training_dir, model_name)
    logging('Loading model..')
    dataset_provider = DatasetPreprocessorAttention(dataset, config, single_caption=True)
    if 'gridTD' == model_type:
        model = ImgCaptioninggridTDAdaptiveModel(config)
    elif 'adaptive' == model_type:
        model = ImgCaptioningAdaptiveAttentionModel(config)
    else:
        raise NotImplementedError('Please specify model_type as gridTD or adaptive')
    model.build(dataset_provider.vocabs, dataset_provider.vocab_size)
    keras_model = model.keras_model
    logging('Loading model weights..')
    keras_model.load_weights(model_path, by_name=True)

    inference = BeamSearchInference(keras_model, dataset=dataset, config=config,
                                    test_dataset=test_dataset, test_config=test_config,
                                    beam_size=beam_size,
                                    max_caption_length=max_caption_length)
    logging('Evaluating {} set..'.format(dataset_type))
    if dataset_type == 'test':
        metrics, save_predictions, save_reference = inference.evaluate_test_set(
                                                    include_prediction=True)
    elif dataset_type == 'train':
        metrics, save_predictions, save_reference = inference.evaluate_training_set(
                                                    include_prediction=True)
    else:
        metrics, save_predictions, save_reference = inference.evaluate_validation_set(
                                                    include_prediction=True)
    print(metrics)
    logging('Writting result to files..')
    # metrics_path = os.path.join(training_dir,
    #         '{}-metrics-{}-{}-{}-{}-colthes_unbalanced.yaml'.format(dataset_type, beam_size, model_name, test_dataset.DATASET_NAME,
    #                                        max_caption_length))
    # predictions_path = os.path.join(training_dir,
    #         '{}-predictions-{}-{}-{}-{}-colthes_unbalanced.yaml'.format(dataset_type, beam_size, model_name, test_dataset.DATASET_NAME,
    #                                            max_caption_length))

    metrics_path = os.path.join(training_dir,
            '{}-metrics-{}-{}-{}-{}.yaml'.format(dataset_type, beam_size, model_name, test_dataset.DATASET_NAME,
                                           max_caption_length))
    save_predictions_path = os.path.join(training_dir,
            '{}-predictions-{}-{}-{}-{}.yaml'.format(dataset_type, beam_size, model_name, test_dataset.DATASET_NAME,
                                               max_caption_length))
    save_reference_path = os.path.join(training_dir,
                                         '{}-reference-{}-{}-{}-{}.yaml'.format(dataset_type, beam_size, model_name,
                                                                                  test_dataset.DATASET_NAME,
                                                                                  max_caption_length))
    write_yaml_file(metrics, metrics_path)
    write_yaml_file(save_predictions, save_predictions_path)
    write_yaml_file(save_reference, save_reference_path)


    logging('Done!')


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # =========for flickr=============
    flickr_config = config.FlickrConfig()
    flickr_config.batch_size=20
    dataset = Flickr30kDataset(flickr_config, single_caption=True)
    training_dir = './results/flickr30k/training-results/flickr_VGG16_adaptive_attention/'
    # training_dir = './results/flickr30k/training-results/flickr_VGG16_gridTD_attention/'
    # training_dir = './results/flickr30k/training-results/flickr_VGG16_adaptive_lrp_inference_0.5_0.5'
    # training_dir = './results/flickr30k/training-results/flickr_VGG16_adaptive_lrp_inference_baseline'
    # training_dir = './results/flickr30k/training-results/flickr_VGG16_gridTD_lrp_inference_0.5_0.5'
    # training_dir = './results/flickr30k/training-results/flickr_VGG16_gridTD_lrp_inference_baseline'
    main(training_dir, dataset, dataset, flickr_config, flickr_config, model_type='adaptive', beam_size=3,
         dataset_type='test', model_name='keras_model.hdf5')



    # ========for coco ================
    # coco_config = config.COCOConfig()
    # coco_config.batch_size = 20
    # dataset = COCODataset(single_caption=True)

    # training_dir = './results/coco/training-results/coco_VGG16_adaptive_attention/'
    # training_dir = './results/coco/training-results/coco_VGG16_adaptive_lrp_inference_0.5_0.5'
    # training_dir = './results/coco/training-results/coco_VGG16_adaptive_lrp_inference_baseline'
    # training_dir = './results/coco/training-results/coco_VGG16_gridTD_attention/'
    # training_dir = './results/coco/training-results/coco_VGG16_gridTD_lrp_inference_0.5_0.5'
    # training_dir = './results/coco/training-results/coco_VGG16_gridTD_lrp_inference_baseline'

    # for training_dir in [
    #                      './results/coco/training-results/coco_VGG16_adaptive_lrp_inference_0.5_0.5',
    #                      './results/coco/training-results/coco_VGG16_adaptive_lrp_inference_baseline',
    #                      './results/coco/training-results/coco_VGG16_gridTD_attention/',
    #                      './results/coco/training-results/coco_VGG16_gridTD_lrp_inference_0.5_0.5',
    #                      './results/coco/training-results/coco_VGG16_gridTD_lrp_inference_baseline']:
    #     if 'adaptive' in training_dir:
    #         model_type = 'adaptive'
    #     else:
    #         model_type = 'gridTD'
    #     main(training_dir, dataset, dataset, coco_config, coco_config, model_type='gridTD', beam_size=3,
    #          dataset_type='test', model_name='keras_model.hdf5')
    #     K.clear_session()