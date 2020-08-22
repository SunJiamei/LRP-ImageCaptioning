from model import *
from preparedataset import *
import sys
import io_utils
from callbacks import (LogLearningRate, LogMetrics, LogTimestamp,
                        StopAfterTimedelta, StopWhenValLossExploding)
from keras.callbacks import (CSVLogger, EarlyStopping, ModelCheckpoint,
                             ReduceLROnPlateau, LearningRateScheduler)
import glob
from keras.utils import generic_utils
import gc
class Training(object):
    def __init__(self,
                 config,
                 dataset,
                 training_label=None,
                 model_weights_path=None,
                 min_delta=1e-3,
                 min_lr=1e-10,
                 log_metrics_period=1,
                 explode_ratio=0.25,
                 explode_patience=20,
                 max_q_size=10,
                 workers=1,
                 verbose=1,
                 ):
        self._training_label = training_label
        self._config = config
        self._epochs = self._config.num_epochs
        self._reduce_lr_factor = self._config.reduce_lr_factor
        self._reduce_lr_patience = self._config.reduce_lr_patience
        self._early_stopping_patience = self._config.early_stopping_patience
        self._model_weights_path = model_weights_path
        self._min_delta = min_delta
        self._min_lr = min_lr
        self._log_metrics_period = log_metrics_period
        self._explode_ratio = explode_ratio
        self._explode_patience = explode_patience
        self._max_q_size = max_q_size
        self._workers = workers
        self._verbose = verbose
        self._dataset = dataset
        self._dataset_provider = DatasetPreprocessorAttention(self._dataset, self._config)
        self._vocab_size = self._dataset_provider.vocab_size
        self._time_limit = self._config.time_limit
        if not ((self._epochs is None) ^ (self._time_limit is None)):
            raise ValueError('Either conf.epochs or conf.time_limit must be '
                             'set, but not both!')

        if self._time_limit:
            self._epochs = sys.maxsize
        self._stop_training = False
        self._init_result_dir()
        self._init_callbacks()
        config.save_config_as_dict(self._result_dir)
    @property
    def keras_model(self):
        return self._model.keras_model

    def scheduler(self, index, lr):
        if (index + 1) % 3 == 0:
            lr = lr * 0.8
        return lr

    def _init_result_dir(self,):
        self._result_dir = os.path.join(self._dataset_provider.training_results_dir, self._training_label)
        io_utils.mkdir_p(self._result_dir)

    def _init_callbacks(self):
        log_lr = LogLearningRate()
        log_ts = LogTimestamp()
        log_metrics = LogMetrics(self._dataset_provider,
                                 period=self._log_metrics_period)

        CSV_FILENAME = 'metrics-log.csv'
        self._csv_filepath = os.path.join(self._result_dir, CSV_FILENAME)
        csv_logger = CSVLogger(filename=self._csv_filepath)

        CHECKPOINT_FILENAME = 'keras_model_{epoch:02d}_{val_cider:.4f}.hdf5'
        self._checkpoint_filepath = os.path.join(self._result_dir, CHECKPOINT_FILENAME)

        model_checkpoint = ModelCheckpoint(filepath=self._checkpoint_filepath,
                                           monitor='val_cider',
                                           mode='max',
                                           save_best_only=False,
                                           save_weights_only=True,
                                           period=1,
                                           verbose=self._verbose)

        # tensorboard = TensorBoard(log_dir=self._result_dir,
        #                           write_graph=False)

        reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                      mode='min',
                                      min_delta=self._min_delta,
                                      factor=self._reduce_lr_factor,
                                      patience=self._reduce_lr_patience,
                                      min_lr=self._min_lr,
                                      verbose=self._verbose)
        lr_schedule = LearningRateScheduler(self.scheduler, verbose=0)
        earling_stopping = EarlyStopping(monitor='val_cider',
                                         mode='max',
                                         min_delta=self._min_delta,
                                         patience=self._early_stopping_patience,
                                         verbose=self._verbose)

        # stop_after = StopAfterTimedelta(timedelta=self._time_limit,
        #                                 verbose=self._verbose)

        # stop_when = StopWhenValLossExploding(ratio=self._explode_ratio,
        #                                      patience=self._explode_patience,
        #                                      verbose=self._verbose)

        # TODO Add LearningRateScheduler. Is it still needed?

        self._callbacks = [log_lr,  # Must be before tensorboard
                           log_metrics,  # Must be before model_checkpoint and
                                         # tensorboard
                           model_checkpoint,
                           # tensorboard,  # Must be before log_ts
                           log_ts,  # Must be before csv_logger
                           csv_logger,
                           reduce_lr]  # Must be after csv_logger
                           # stop_when,  # Must be the third last
                           #earling_stopping,]  # Must be the second last
                           # stop_after]  # Must be the last

    def run(self):
        io_utils.logging('Building model..')
        self._model.build(self._dataset_provider.vocabs, self._dataset_provider.vocab_size)
        if self._model_weights_path:
            io_utils.logging('Loading model weights from {}..'.format(
                                                    self._model_weights_path))
            self.keras_model.load_weights(self._model_weights_path)


        if self._stop_training:
            self._stop_training = False
            return

        io_utils.logging('Training {} is starting..'.format(self._training_label))

        self.keras_model.fit_generator(
                generator=self._dataset_provider.training_set(),
                steps_per_epoch=self._dataset_provider.training_steps,
                epochs=self._epochs,
                validation_data=self._dataset_provider.validation_set(),
                validation_steps=self._dataset_provider.validation_steps,
                max_queue_size=self._max_q_size,
                workers=self._workers,
                callbacks=self._callbacks,
                verbose=self._verbose)

        self._stop_training = False
        io_utils.logging('Training {} has finished.'.format(self._training_label))

    def stop_training(self):
        self._stop_training = True
        try:
            self.keras_model.stop_training = True
        # Race condition: ImageCaptioningModel.build is not called yet
        except AttributeError:
            pass


class TrainingAdaptiveAttention(Training):
    def __init__(self,
                 config,
                 dataset,
                 training_label=None,
                 model_weights_path=None,
                 min_delta=1e-3,
                 min_lr=1e-6,
                 log_metrics_period=1,
                 explode_ratio=0.25,
                 explode_patience=20,
                 max_q_size=10,
                 workers=1,
                 verbose=1,
                 ):
        super(TrainingAdaptiveAttention, self).__init__(config,
                                                        dataset,
                                                        training_label=training_label,
                                                        model_weights_path=model_weights_path,
                                                        min_delta=min_delta,
                                                        min_lr=min_lr,
                                                        log_metrics_period=log_metrics_period,
                                                        explode_ratio=explode_ratio,
                                                        explode_patience=explode_patience,
                                                        max_q_size=max_q_size,
                                                        workers=workers,
                                                        verbose=verbose)
        self._model = ImgCaptioningAdaptiveAttentionModel(self._config)
        config.save_config_as_dict(self._result_dir)


    def scheduler(self, index, lr):
        if lr <= 1e-5:
            return lr
        if (index + 3) % 4 == 0:
            lr = lr * 0.8
        return lr

    def _init_callbacks(self):
        log_lr = LogLearningRate()
        log_ts = LogTimestamp()
        log_metrics = LogMetrics(self._dataset_provider,
                                 period=self._log_metrics_period)

        CSV_FILENAME = 'metrics-log.csv'
        self._csv_filepath = os.path.join(self._result_dir, CSV_FILENAME)
        csv_logger = CSVLogger(filename=self._csv_filepath)

        CHECKPOINT_FILENAME = 'keras_model_{epoch:02d}_{val_cider:.4f}.hdf5'
        self._checkpoint_filepath = os.path.join(self._result_dir, CHECKPOINT_FILENAME)

        model_checkpoint = ModelCheckpoint(filepath=self._checkpoint_filepath,
                                           monitor='val_loss',
                                           mode='min',
                                           save_best_only=False,
                                           save_weights_only=True,
                                           period=1,
                                           verbose=self._verbose)

        reduce_lr = ReduceLROnPlateau(monitor='val_cider',
                                      mode='max',
                                      min_delta=self._min_delta,
                                      factor=self._reduce_lr_factor,
                                      patience=self._reduce_lr_patience,
                                      min_lr=self._min_lr,
                                      verbose=self._verbose)
        lr_schedule = LearningRateScheduler(self.scheduler, verbose=0)
        earling_stopping = EarlyStopping(monitor='val_cider',
                                         mode='max',
                                         min_delta=self._min_delta,
                                         patience=self._early_stopping_patience,
                                         verbose=self._verbose)

        stop_after = StopAfterTimedelta(timedelta=self._time_limit,
                                        verbose=self._verbose)

        stop_when = StopWhenValLossExploding(ratio=self._explode_ratio,
                                             patience=self._explode_patience,
                                             verbose=self._verbose)

        # TODO Add LearningRateScheduler. Is it still needed?

        self._callbacks = [log_lr,  # Must be before tensorboard
                           log_metrics,  # Must be before model_checkpoint and tensorboard
                           model_checkpoint,
                           log_ts,  # Must be before csv_logger
                           csv_logger,
                           reduce_lr]  # Must be after csv_logger
                           # stop_when,  # Must be the third last
                           #earling_stopping,]  # Must be the second last
                           # stop_after]  # Must be the last

    def run(self):
        io_utils.io_utils.logging('Building model..')
        self._model.build(self._dataset_provider.vocabs, self._dataset_provider.vocab_size)
        if self._model_weights_path:
            io_utils.io_utils.logging('Loading model weights from {}..'.format(
                                                    self._model_weights_path))
            self.keras_model.load_weights(self._model_weights_path)

        if self._stop_training:
            self._stop_training = False
            return

        io_utils.io_utils.logging('Training {} is starting..'.format(self._training_label))

        self.keras_model.fit_generator(
                generator=self._dataset_provider.training_set(),
                steps_per_epoch=self._dataset_provider.training_steps,
                epochs=self._epochs,
                validation_data=self._dataset_provider.test_set(),
                validation_steps=self._dataset_provider.test_steps,
                max_queue_size=self._max_q_size,
                workers=self._workers,
                callbacks=self._callbacks,
                verbose=self._verbose)

        self._stop_training = False
        io_utils.io_utils.logging('Training {} has finished.'.format(self._training_label))


class TrainingGridTD(Training):
    def __init__(self,
                 config,
                 dataset,
                 training_label=None,
                 model_weights_path=None,
                 min_delta=1e-3,
                 min_lr=1e-7,
                 log_metrics_period=1,
                 explode_ratio=0.25,
                 explode_patience=20,
                 max_q_size=10,
                 workers=1,
                 verbose=1,
                 ):
        super(TrainingGridTD, self).__init__(config,
                                             dataset,
                                             training_label=training_label,
                                             model_weights_path=model_weights_path,
                                             min_delta=min_delta,
                                             min_lr=min_lr,
                                             log_metrics_period=log_metrics_period,
                                             explode_ratio=explode_ratio,
                                             explode_patience=explode_patience,
                                             max_q_size=max_q_size,
                                             workers=workers,
                                             verbose=verbose)
        self._model = ImgCaptioninggridTDAdaptiveModel(self._config)
        config.save_config_as_dict(self._result_dir)

    def scheduler(self, index, lr):
        if lr <= 1e-5:
            return lr
        if (index + 3) % 4 == 0:
            lr = lr * 0.8
        return lr

    def _init_callbacks(self):
        log_lr = LogLearningRate()
        log_ts = LogTimestamp()
        log_metrics = LogMetrics(self._dataset_provider,
                                 period=self._log_metrics_period)

        CSV_FILENAME = 'metrics-log.csv'
        self._csv_filepath = os.path.join(self._result_dir, CSV_FILENAME)
        csv_logger = CSVLogger(filename=self._csv_filepath)

        CHECKPOINT_FILENAME = 'keras_model_{epoch:02d}_{val_cider:.4f}.hdf5'
        self._checkpoint_filepath = os.path.join(self._result_dir, CHECKPOINT_FILENAME)

        model_checkpoint = ModelCheckpoint(filepath=self._checkpoint_filepath,
                                           monitor='val_loss',
                                           mode='min',
                                           save_best_only=False,
                                           save_weights_only=True,
                                           period=1,
                                           verbose=self._verbose)

        reduce_lr = ReduceLROnPlateau(monitor='val_cider',
                                      mode='max',
                                      min_delta=self._min_delta,
                                      factor=self._reduce_lr_factor,
                                      patience=self._reduce_lr_patience,
                                      min_lr=self._min_lr,
                                      verbose=self._verbose)
        lr_schedule = LearningRateScheduler(self.scheduler, verbose=0)
        earling_stopping = EarlyStopping(monitor='val_cider',
                                         mode='max',
                                         min_delta=self._min_delta,
                                         patience=self._early_stopping_patience,
                                         verbose=self._verbose)

        # stop_after = StopAfterTimedelta(timedelta=self._time_limit,
        #                                 verbose=self._verbose)

        # stop_when = StopWhenValLossExploding(ratio=self._explode_ratio,
        #                                      patience=self._explode_patience,
        #                                      verbose=self._verbose)

        # TODO Add LearningRateScheduler. Is it still needed?

        self._callbacks = [log_lr,  # Must be before tensorboard
                           log_metrics,  # Must be before model_checkpoint and tensorboard
                           model_checkpoint,
                           log_ts,  # Must be before csv_logger
                           csv_logger,
                           reduce_lr]  # Must be after csv_logger

    def run(self):
        io_utils.io_utils.logging('Building model..')
        self._model.build(self._dataset_provider.vocabs, self._dataset_provider.vocab_size)
        if self._model_weights_path:
            io_utils.io_utils.logging('Loading model weights from {}..'.format(
                                                    self._model_weights_path))
            self.keras_model.load_weights(self._model_weights_path)

        if self._stop_training:
            self._stop_training = False
            return

        io_utils.io_utils.logging('Training {} is starting..'.format(self._training_label))

        self.keras_model.fit_generator(
                generator=self._dataset_provider.training_set(),
                steps_per_epoch=self._dataset_provider.training_steps,
                epochs=self._epochs,
                validation_data=self._dataset_provider.test_set(),
                validation_steps=self._dataset_provider.test_steps,
                max_queue_size=self._max_q_size,
                workers=self._workers,
                callbacks=self._callbacks,
                verbose=self._verbose)
        self._stop_training = False
        io_utils.io_utils.logging('Training {} has finished.'.format(self._training_label))


class TrainingLRPInferenceAdaptiveAttention(Training):
    def __init__(self,
                 config,
                 dataset,
                 training_label=None,
                 model_weights_path=None,
                 min_delta=1e-4,
                 min_lr=1e-10,
                 log_metrics_period=1,
                 explode_ratio=0.25,
                 explode_patience=20,
                 max_q_size=10,
                 workers=1,
                 verbose=1,
                 ):
        super(TrainingLRPInferenceAdaptiveAttention, self).__init__(config,
                                                                    dataset,
                                                                    training_label=training_label,
                                                                    model_weights_path=model_weights_path,
                                                                    min_delta=min_delta,
                                                                    min_lr=min_lr,
                                                                    log_metrics_period=log_metrics_period,
                                                                    explode_ratio=explode_ratio,
                                                                    explode_patience=explode_patience,
                                                                    max_q_size=max_q_size,
                                                                    workers=workers,
                                                                    verbose=verbose)
        self.T = config.sentence_length
        self.L = config.img_feature_length
        self.D = config.img_feature_dim
        self._model = ImgCaptioningAdaptiveAttentionLRPInferenceModel(self._config, self._dataset_provider)
        config.save_config_as_dict(self._result_dir)

    def run(self, save_idx, epoch_length):
        io_utils.io_utils.logging('Building model..')
        self._model.build(self._dataset_provider.vocabs, self._dataset_provider.vocab_size)
        if self._model_weights_path:
            io_utils.io_utils.logging('Loading model weights from {}..'.format(
                                                    self._model_weights_path))
            self.keras_model.load_weights(self._model_weights_path)

        io_utils.io_utils.logging('Training {} is starting..'.format(self._training_label))
        total_loss = np.zeros((epoch_length, 3))
        dataset_loader = self._dataset_provider.training_set(include_datum=False)
        for i in range(save_idx * epoch_length):
            previous_data = next(dataset_loader)
        for epoch_num in range(self._epochs):
            progbar = generic_utils.Progbar(epoch_length)
            print('Epoch {}/{}'.format(epoch_num, self._epochs))
            iter_num = 0
            while True:
                X, y = next(dataset_loader)
                y_pred = self.keras_model.predict_on_batch(X+[np.zeros(y.shape)])[0]
                lrp_model = LRPInferenceLayerAdaptive(self.keras_model, self._dataset_provider, self._model._hidden_dim,
                                                      self._model._embedding_dim, self.L, self.D, self._model.img_encoder, 'mean')
                lrp_weight = lrp_model.call(X+[y_pred])
                del lrp_model
                gc.collect()
                losses = self.keras_model.train_on_batch(X+[lrp_weight], [y,y])
                acc = losses[3]

                total_loss[iter_num,0] = losses[0]
                total_loss[iter_num, 1] = losses[1]
                total_loss[iter_num,2] = acc
                iter_num += 1
                progbar.update(iter_num,
                               [('loss', np.mean(total_loss[: iter_num, 1])), ('acc', np.mean(total_loss[: iter_num,2]))])

                if iter_num == epoch_length:
                    self.keras_model.save_weights(
                        os.path.join(self._result_dir, 'keras_model_{:02d}_{:.4f}.hdf5'.format(save_idx, acc)))
                    print('loss: ', np.mean(total_loss, axis=0)[0], np.mean(total_loss, axis=0)[1])
                    print('acc: ', np.mean(total_loss, axis=0)[2])
                    break
        io_utils.io_utils.logging('Training {} has finished.'.format(self._training_label))


class TrainingLRPInferenceGridTD(Training):
    def __init__(self,
                 config,
                 dataset,
                 training_label=None,
                 model_weights_path=None,
                 min_delta=1e-4,
                 min_lr=1e-10,
                 log_metrics_period=1,
                 explode_ratio=0.25,
                 explode_patience=20,
                 max_q_size=10,
                 workers=1,
                 verbose=1,
                 ):
        super(TrainingLRPInferenceGridTD, self).__init__(config,
                                                         dataset,
                                                         training_label=training_label,
                                                         model_weights_path=model_weights_path,
                                                         min_delta=min_delta-4,
                                                         min_lr=min_lr,
                                                         log_metrics_period=log_metrics_period,
                                                         explode_ratio=explode_ratio,
                                                         explode_patience=explode_patience,
                                                         max_q_size=max_q_size,
                                                         workers=workers,
                                                         verbose=verbose)

        self._model = ImgCaptioningGridTDLRPInferenceModel(self._config, self._dataset_provider)
        self.T = config.sentence_length
        self.L = config.img_feature_length
        self.D = config.img_feature_dim
        config.save_config_as_dict(self._result_dir)

    def run(self, save_idx, epoch_length):
        # epoch_length = 20
        io_utils.io_utils.logging('Building model..')
        self._model.build(self._dataset_provider.vocabs, self._dataset_provider.vocab_size)
        if self._model_weights_path:
            io_utils.io_utils.logging('Loading model weights from {}..'.format(
                                                    self._model_weights_path))
            self.keras_model.load_weights(self._model_weights_path)

        io_utils.io_utils.logging('Training {} is starting..'.format(self._training_label))
        total_loss = np.zeros((epoch_length, 3))
        dataset_loader = self._dataset_provider.training_set(include_datum=False)
        for i in range(save_idx * epoch_length):
            previous_data = next(dataset_loader)
        for epoch_num in range(self._epochs):
            progbar = generic_utils.Progbar(epoch_length)
            print('Epoch {}/{}'.format(epoch_num, self._epochs))
            iter_num = 0
            while True:
                X, y = next(dataset_loader)
                print(X[0].shape, y.shape)
                y_pred = self.keras_model.predict_on_batch(X+[np.zeros(y.shape)])[0]
                lrp_model = LRPInferenceLayergridTD(self.keras_model, self._dataset_provider, self._model._hidden_dim,
                                                    self._model._embedding_dim, self.L, self.D, self._model.img_encoder, 'mean')
                lrp_weight = lrp_model.call(X+[y_pred])
                del lrp_model
                gc.collect()
                losses = self.keras_model.train_on_batch(X+[lrp_weight], [y,y])
                acc = losses[3]

                total_loss[iter_num,0] = losses[0]
                total_loss[iter_num, 1] = losses[1]
                total_loss[iter_num,2] = acc
                iter_num += 1
                progbar.update(iter_num,
                               [('loss', np.mean(total_loss[: iter_num, 1])), ('acc', np.mean(total_loss[: iter_num,2]))])

                if iter_num == epoch_length:
                    self.keras_model.save_weights(
                        os.path.join(self._result_dir, 'keras_model_{:02d}_{:.4f}.hdf5'.format(save_idx, acc)))
                    print('loss: ', np.mean(total_loss, axis=0)[0], np.mean(total_loss, axis=0)[1])
                    print('acc: ', np.mean(total_loss, axis=0)[2])
                    break
        io_utils.logging('Training {} has finished.'.format(self._training_label))


MODELTYPE = {'adaptiveattention':TrainingAdaptiveAttention, 'gridTD':TrainingGridTD}


def main_attention(config, dataset, training_label, model_type,log_metrics_period=1):
    training = MODELTYPE[model_type](config, dataset, training_label=training_label, log_metrics_period=log_metrics_period)
    training.run()



if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # to train from scratch
    flickr_config = config.FlickrConfig()
    flickr_config.learning_rate = 1e-6
    flickr_config.num_epochs = 1
    flickr_config.batch_size = 32
    dataset = Flickr30kDataset(flickr_config)
    main_attention(flickr_config,dataset,'flickr_VGG16_adaptive_attention/', 'adaptiveattention')


    # to finetune with LRP inference
    flickr_config = config.FlickrConfig()
    flickr_config.learning_rate = 1e-6
    flickr_config.num_epochs = 1
    flickr_config.batch_size = 32
    dataset = Flickr30kDataset(flickr_config)
    for i in range(50):
        if i == 0:
            model_weight = glob.glob('./results/flickr30k/training-results/flickr_VGG16_adaptive_attention/keras_model.hdf5')[0]
            training = TrainingLRPInferenceGridTD(config=flickr_config, dataset=dataset,
                                                  training_label='flickr_VGG16_adaptive_attention_lrp_inference_real_time_0.5_0.5_test',
                                                  model_weights_path=model_weight)
            training.run(i+1, 10)
        else:
            model_weight = glob.glob(
                './results/flickr30k/training-results/flickr_VGG16_adaptive_attention_lrp_inference_real_time_0.5_0.5_test/keras_model_{:02d}*'.format(i))[0]
            training = TrainingLRPInferenceGridTD(config=flickr_config, dataset=dataset,
                                                  training_label='flickr_VGG16_adaptive_attention_lrp_inference_real_time_0.5_0.5_test',
                                                  model_weights_path=model_weight)
            training.run(i+1, 10)
        K.clear_session()


