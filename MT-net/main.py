"""
Usage Instructions:
    Scripts with hyperparameters are in experiments/

    To run evaluation, use the '--train=False' flag and the '--test_set=True' flag to use the test set.
"""

import csv
import numpy as np
import pickle
import random
import tensorflow as tf
import pdb

#from data_generator import DataGenerator
#from poly_generator import PolyDataGenerator
from maml import MAML
from data_generator_ti import TieredGenerator
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

## Dataset/method options
flags.DEFINE_string('datasource', 'sinusoid', 'sinusoid or omniglot or miniimagenet')
flags.DEFINE_integer('num_classes', 5, 'number of classes used in classification (e.g. 5-way classification).')
flags.DEFINE_integer('num_train_classes', -1, 'number of classes to train on (-1 for all).')
# oracle means task id is input (only suitable for sinusoid)
flags.DEFINE_string('baseline', None, 'oracle, or None')

## Training options
flags.DEFINE_integer('pretrain_iterations', 0, 'number of pre-training iterations.')
flags.DEFINE_integer('metatrain_iterations', 15000, 'number of metatraining iterations.') # 15k for omniglot, 50k for sinusoid
flags.DEFINE_integer('meta_batch_size', 25, 'number of tasks sampled per meta-update')
flags.DEFINE_float('meta_lr', 1e-3, 'the base learning rate of the generator')
flags.DEFINE_integer('update_batch_size', 5, 'number of examples used for inner gradient update (K for K-shot learning).')
flags.DEFINE_float('update_lr', 3e-2, 'step size alpha for inner gradient update.') # 0.1 for omniglot
flags.DEFINE_integer('num_updates', 1, 'number of inner gradient updates during training.')
flags.DEFINE_integer('poly_order', 1, 'order of polynomial to generate')
flags.DEFINE_integer('kshot', 0, 'if 0: anyshot train / k: maximum kshot train')

## Model options
#flags.DEFINE_string('mod', '', 'modifications to original paper. None, split, both')
flags.DEFINE_bool('use_T', False, 'whether or not to use transformation matrix T')
flags.DEFINE_bool('use_M', False, 'whether or not to use mask M')
flags.DEFINE_bool('share_M', False, 'only effective if use_M is true, whether or not to '
                                    'share masks between weights'
                                    'that contribute to the same activation')
flags.DEFINE_float('temp', 1, 'temperature for gumbel-softmax')
flags.DEFINE_float('logit_init', 0, 'initial logit')
flags.DEFINE_string('norm', 'batch_norm', 'batch_norm, layer_norm, or None')
flags.DEFINE_integer('dim_hidden', 40, 'dimension of fc layer')
flags.DEFINE_integer('num_filters', 64, 'number of filters for conv nets -- use 32 for '
                                        'miniimagenet, 64 for omiglot.')
flags.DEFINE_bool('conv', True, 'whether or not to use a convolutional network, only applicable in some cases')
flags.DEFINE_bool('max_pool', False, 'Whether or not to use max pooling rather than strided convolutions')
flags.DEFINE_bool('stop_grad', False, 'if True, do not use second derivatives in meta-optimization (for speed)')

## Logging, saving, and testing options
flags.DEFINE_bool('log', True, 'if false, do not log summaries, for debugging code.')
flags.DEFINE_string('logdir', '/tmp/data', 'directory for summaries and checkpoints.')
flags.DEFINE_bool('debug', False, 'debug mode. uses less data for fast evaluation.')
flags.DEFINE_bool('resume', True, 'resume training if there is a model available')
flags.DEFINE_bool('train', True, 'True to train, False to test.')
flags.DEFINE_integer('test_iter', -1, 'iteration to load model (-1 for latest model)')
flags.DEFINE_bool('test_set', False, 'Set to true to test on the the test set, False for the validation set.')
flags.DEFINE_integer('train_update_batch_size', -1, 'number of examples used for gradient update during training (use if you want to test with a different number).')
flags.DEFINE_float('train_update_lr', -1, 'value of inner gradient step step during training. (use if you want to test with a different value)') # 0.1 for omniglot


def train(model, saver, sess, exp_string, data_generator, resume_itr=0):
    SUMMARY_INTERVAL = 100
    SAVE_INTERVAL = 5000
    if FLAGS.debug:
        SUMMARY_INTERVAL = PRINT_INTERVAL = 10
        TEST_PRINT_INTERVAL = PRINT_INTERVAL*5
    elif FLAGS.datasource in ['sinusoid', 'polynomial']:
        PRINT_INTERVAL = 1000
        TEST_PRINT_INTERVAL = PRINT_INTERVAL*5
    else:
        PRINT_INTERVAL = 100
        TEST_PRINT_INTERVAL = PRINT_INTERVAL*5

    if FLAGS.log:
        train_writer = tf.summary.FileWriter(FLAGS.logdir + '/' + exp_string, sess.graph)
    print('Done initializing, starting training.')
    prelosses, postlosses = [], []

    num_classes = FLAGS.num_classes # for classification, 1 otherwise
    multitask_weights, reg_weights = [], []

    for itr in range(resume_itr+1, 
            FLAGS.pretrain_iterations + FLAGS.metatrain_iterations+1):
        xa, xb, ya, yb = data_generator.data_queue('train', FLAGS.meta_batch_size, num_classes, FLAGS.kshot)
        feed_dict = {model.inputa: xa, model.inputb: xb, model.labela: ya, model.labelb: yb}

        input_tensors = [model.metatrain_op]
#        input_tensors.extend([model.summ_op, model.total_loss1,
#                              model.total_losses2[FLAGS.num_updates-1]])
#        if model.classification:
        input_tensors.extend([model.total_accuracy1, model.total_accuracies2[FLAGS.num_updates-1]])
        result = sess.run(input_tensors, feed_dict)

        prelosses.append(result[-2])
        postlosses.append(result[-1])

        if itr != 0 and itr % PRINT_INTERVAL == 0:
            if itr < FLAGS.pretrain_iterations:
                print_str = 'Pretrain Iteration ' + str(itr)
            else:
                print_str = 'Iteration ' + str(itr - FLAGS.pretrain_iterations)
            print_str += ': ' + str(np.mean(prelosses)) + ', ' + str(np.mean(postlosses))
            print(print_str)
            #print sess.run(model.total_probs)
            prelosses, postlosses = [], []

        if itr != 0 and itr % SAVE_INTERVAL == 0:
            saver.save(sess, FLAGS.logdir + '/' + exp_string + '/model' + str(itr))
            test(model, saver, sess, exp_string, data_generator, FLAGS.num_updates, 
                    mode='val')

        # sinusoid is infinite data, so no need to test on meta-validation set.
#        if itr != 0 and itr % TEST_PRINT_INTERVAL == 0 and FLAGS.datasource not in ['sinusoid', 'polynomial']:
#            if 'generate' not in dir(data_generator):
#                feed_dict = {}
#                if model.classification:
#                    input_tensors = [model.metaval_total_accuracy1,
#                                     model.metaval_total_accuracies2[FLAGS.num_updates-1], model.summ_op]
#                else:
#                    input_tensors = [model.metaval_total_loss1,
#                                     model.metaval_total_losses2[FLAGS.num_updates-1], model.summ_op]
#            else:
#                batch_x, batch_y, amp, phase = data_generator.generate(train=False)
#                inputa = batch_x[:, :num_classes*FLAGS.update_batch_size, :]
#                inputb = batch_x[:, num_classes*FLAGS.update_batch_size:, :]
#                labela = batch_y[:, :num_classes*FLAGS.update_batch_size, :]
#                labelb = batch_y[:, num_classes*FLAGS.update_batch_size:, :]
#                feed_dict = {model.inputa: inputa, model.inputb: inputb,
#                             model.labela: labela, model.labelb: labelb, model.meta_lr: 0.0}
#                if model.classification:
#                    input_tensors = [model.total_accuracy1, model.total_accuracies2[FLAGS.num_updates-1]]
#                else:
#                    input_tensors = [model.total_loss1, model.total_losses2[FLAGS.num_updates-1]]
#
#            result = sess.run(input_tensors, feed_dict)
#            print('Validation results: ' + str(result[0]) + ', ' + str(result[1]))

    saver.save(sess, FLAGS.logdir + '/' + exp_string + '/model' + str(itr))


def test(model, saver, sess, exp_string, data_generator, test_num_updates=None, mode='test'):
    num_classes = FLAGS.num_classes # for classification, 1 otherwise
#    np.random.seed(1)
#    random.seed(1)
    metaval_accuracies = []
    print_test_class_acc = False
    
    per_class_acc = np.zeros([len(data_generator.get_dataset(mode)), 2])
    per_class_avg = []
    NUM_TEST_POINTS = 600
    for point_n in range(NUM_TEST_POINTS):
        xa, xb, ya, yb, task_ind = data_generator.data_queue(mode, FLAGS.meta_batch_size, num_classes, debug=True)
        feed_dict = {model.inputa: xa, model.inputb: xb, model.labela: ya, model.labelb: yb}
        result = sess.run([model.metaval_total_accuracy1] + model.metaval_total_accuracies2, feed_dict)
        metaval_accuracies.append(result)

        if print_test_class_acc:
            # get per class accuracy
            preds = sess.run(model.outputbs, feed_dict)
            p = np.argmax(preds[FLAGS.num_updates-1][0], 1)
            y = np.argmax(yb[0], 1)
            right = (p == y)

            tmp_result = [np.mean(right[y==n]) for n in range(num_classes)]
            per_class_avg.append(np.mean(tmp_result))
            
            per_class_acc[task_ind, 1] += 1
            per_class_acc[task_ind[right], 0] += 1
    
    np.set_printoptions(4)
    if print_test_class_acc:
        print ('classwise accuracy')
        print (per_class_acc[:, 0] \
                / per_class_acc[:, 1] * 100)

        print ('per class avg accuracy')
        print ('  {:.4f} +- {:.4f}'.format(np.mean(per_class_avg)*100., 
            np.std(per_class_avg)*100.*1.96/np.sqrt(NUM_TEST_POINTS)))

    metaval_accuracies = np.array(metaval_accuracies)
    means = np.mean(metaval_accuracies, 0)
    stds = np.std(metaval_accuracies, 0)
    ci95 = 1.96*stds/np.sqrt(NUM_TEST_POINTS)

    print('Mean validation accuracy/loss, stddev, and confidence intervals')
    print('   {:.4f} +- {:.4f}'.format(means[-1]*100., ci95[-1]*100.))
    filename = FLAGS.logdir + '/' + exp_string + '/' + 'test_ubs' + str(FLAGS.update_batch_size) + \
               '_stepsize' + str(FLAGS.update_lr) + '_testiter' + str(FLAGS.test_iter)
    print (filename)
#    with open(filename + '.pkl', 'w') as f:
#        pickle.dump({'mses': metaval_accuracies}, f)
#    with open(filename + '.csv', 'w') as f:
#        writer = csv.writer(f, delimiter=',')
#        writer.writerow(['update'+str(i) for i in range(len(means))])
#        writer.writerow(means)
#        writer.writerow(stds)
#        writer.writerow(ci95)

def main():
    if FLAGS.datasource in ['sinusoid', 'polynomial']:
        if FLAGS.train:
            test_num_updates = 5
        else:
            test_num_updates = 10
    elif FLAGS.datasource == 'miniimagenet':
        if FLAGS.train:
            test_num_updates = 1  # eval on at least one update during training
        else:
            test_num_updates = FLAGS.num_updates
    else:
        test_num_updates = FLAGS.num_updates

    if not FLAGS.train:
        orig_meta_batch_size = FLAGS.meta_batch_size
        # always use meta batch size of 1 when testing.
        FLAGS.meta_batch_size = 1
    
    data_generator = TieredGenerator(seed=0)


    dim_input = 84*84*3
    dim_output = FLAGS.num_classes

    if FLAGS.datasource == 'miniimagenet' or FLAGS.datasource == 'omniglot':
        tf_data_load = True
        inputa = tf.placeholder(tf.float32, 
               [FLAGS.meta_batch_size,None,dim_input])
        inputb = tf.placeholder(tf.float32, 
               [FLAGS.meta_batch_size,None,dim_input])
        labela = tf.placeholder(tf.int32, 
               [FLAGS.meta_batch_size,None,dim_output])
        labelb = tf.placeholder(tf.int32, 
                [FLAGS.meta_batch_size,None,dim_output])

        if FLAGS.train: # only construct training model if needed
            input_tensors = {'inputa': inputa, 'inputb': inputb, 'labela': labela, 'labelb': labelb}
        metaval_input_tensors = {'inputa': inputa, 'inputb': inputb, 'labela': labela, 'labelb': labelb}

    model = MAML(dim_input, dim_output, test_num_updates=test_num_updates)
    if FLAGS.train or not tf_data_load:
        model.construct_model(input_tensors=input_tensors, prefix='metatrain_')
    if tf_data_load:
        model.construct_model(input_tensors=metaval_input_tensors, prefix='metaval_')
    model.summ_op = tf.summary.merge_all()

    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=10)

    sess = tf.InteractiveSession()

    if not FLAGS.train:
        # change to original meta batch size when loading model.
        FLAGS.meta_batch_size = orig_meta_batch_size

    if FLAGS.train_update_batch_size == -1:
        FLAGS.train_update_batch_size = FLAGS.update_batch_size
    if FLAGS.train_update_lr == -1:
        FLAGS.train_update_lr = FLAGS.update_lr

    exp_string = 'cls_'+str(FLAGS.num_classes)+\
                 '.mbs_'+str(FLAGS.meta_batch_size) + \
                 '.ubs_' + str(FLAGS.train_update_batch_size) + \
                 '.numstep' + str(FLAGS.num_updates) + \
                 '.updatelr' + str(FLAGS.train_update_lr) + \
                 '.temp' + str(FLAGS.temp)

    if FLAGS.debug:
        exp_string += '!DEBUG!'

    if FLAGS.use_T and FLAGS.use_M and FLAGS.share_M:
        exp_string += 'MTnet'
    if FLAGS.use_T and not FLAGS.use_M:
        exp_string += 'Tnet'
    if not FLAGS.use_T and FLAGS.use_M and FLAGS.share_M:
        exp_string += 'Mnet'
    if FLAGS.use_T and FLAGS.use_M and not FLAGS.share_M:
        exp_string += 'MTnet_noshare'
    if not FLAGS.use_T and FLAGS.use_M and not FLAGS.share_M:
        exp_string += 'Mnet_noshare'
    if not FLAGS.use_T and not FLAGS.use_M:
        exp_string += 'MAML'

    if FLAGS.datasource == 'polynomial':
        exp_string += 'ord' + str(FLAGS.poly_order)
    if FLAGS.num_train_classes != -1:
        exp_string += 'ntc' + str(FLAGS.num_train_classes)
    if FLAGS.num_filters != 64:
        exp_string += 'hidden' + str(FLAGS.num_filters)
    if FLAGS.max_pool:
        exp_string += 'maxpool'
    if FLAGS.stop_grad:
        exp_string += 'stopgrad'
    if FLAGS.baseline:
        exp_string += FLAGS.baseline
    if FLAGS.kshot:
        exp_string += 'k{}'.format(FLAGS.kshot)
    if FLAGS.norm == 'batch_norm':
        exp_string += 'batchnorm'
    elif FLAGS.norm == 'layer_norm':
        exp_string += 'layernorm'
    elif FLAGS.norm == 'None':
        exp_string += 'nonorm'
    else:
        print('Norm setting not recognized.')

    resume_itr = 0
    tf.global_variables_initializer().run()
    #tf.train.start_queue_runners()

    if FLAGS.resume or not FLAGS.train:
        model_file = tf.train.latest_checkpoint(FLAGS.logdir + '/' + exp_string)
        # if you want to use other:
        # model_file = model_file.replace('59999', '35000')
        if FLAGS.test_iter > 0:
            model_file = model_file[:model_file.index('model')] + 'model' + str(FLAGS.test_iter)
        if model_file:
            ind1 = model_file.index('model')
            resume_itr = int(model_file[ind1+5:])
            print("Restoring model weights from " + model_file)
            saver.restore(sess, model_file)
    
    for key in FLAGS.__flags.keys():
        print ('{:20s}  :  {}'.format(key, getattr(FLAGS, key)))
    print (exp_string)

    if FLAGS.train:
        train(model, saver, sess, exp_string, data_generator, resume_itr)
    else:
        test(model, saver, sess, exp_string, data_generator, test_num_updates, mode='val')


if __name__ == "__main__":
    main()
