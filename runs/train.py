import os.path
import shutil
import sys
import numpy as np

import tensorflow as tf

from nets import nets
from util import summary


def run(opt):

    ################################################################################################
    # Read experiment to run
    ################################################################################################

    # Skip execution if instructed in experiment
    if opt.skip:
        print("SKIP")
        quit()

    print(opt.name)
    print("INIT ", opt.hyper.init_factor)
    print("Max epochs ", opt.hyper.max_num_epochs)
    print("lr ", opt.hyper.learning_rate)
    print("alpha ", opt.hyper.alpha)
    print("batch size ", opt.hyper.batch_size)
    # print("Iterations: {}".format(opt.dnn.n_t))
    print("Training: {}".format(opt.dataset.type))

    ################################################################################################
    # Define training and validation datasets through Dataset API
    ################################################################################################

    # Initialize dataset and creates TF records if they do not exist

    if opt.dataset.dataset_name == 'symmetry':
        from data import symmetry_data
        dataset = symmetry_data.SymmetryDataset(opt)
    else:
        print("Error: no valid dataset specified")

    # Repeatable datasets for training
    train_dataset = dataset.create_dataset(augmentation=False, standarization=False, set_name='train', repeat=True)
    val_dataset = dataset.create_dataset(augmentation=False, standarization=False, set_name='val', repeat=True)

    # No repeatable dataset for testing
    train_dataset_full = dataset.create_dataset(augmentation=False, standarization=False, set_name='train', repeat=True)
    val_dataset_full = dataset.create_dataset(augmentation=False, standarization=False, set_name='val', repeat=True)
    test_dataset_full = dataset.create_dataset(augmentation=False, standarization=False, set_name='test', repeat=True)

    # Handles to switch datasets
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(
        handle, train_dataset.output_types, train_dataset.output_shapes)

    train_iterator = train_dataset.make_one_shot_iterator()
    val_iterator = val_dataset.make_one_shot_iterator()

    train_iterator_full = train_dataset_full.make_initializable_iterator()
    val_iterator_full = val_dataset_full.make_initializable_iterator()
    test_iterator_full = test_dataset_full.make_initializable_iterator()
    ################################################################################################


    ################################################################################################
    # Declare DNN
    ################################################################################################

    # Get data from dataset dataset
    image, y_ = iterator.get_next()
    y_ = tf.reshape(tensor=y_, shape=[opt.hyper.batch_size])

    # Call DNN
    dropout_rate = tf.placeholder(tf.float32)
    to_call = getattr(nets, opt.dnn.name)
    y, parameters, activations = to_call(image, opt, dropout_rate, len(dataset.list_labels)*dataset.num_outputs)

    print("\n\nUSING NETWORK:", opt.dnn.name)
    print("\n\nUSING DATASET:", dataset.categories)

    # Loss
    with tf.name_scope('loss'):
        weights_norm = tf.reduce_sum(
            input_tensor=opt.hyper.weight_decay * tf.stack(
                [tf.nn.l2_loss(i) for i in parameters]
            ),
            name='weights_norm')
        tf.summary.scalar('weight_decay', weights_norm)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=y)
        cross_entropy_sum = tf.reduce_sum(cross_entropy)
        tf.summary.scalar('cross_entropy', cross_entropy_sum)
        total_loss = weights_norm + cross_entropy_sum
        tf.summary.scalar('total_loss', total_loss)

    global_step = tf.Variable(0, name='global_step', trainable=False)
    ################################################################################################


    ################################################################################################
    # Set up Training
    ################################################################################################

    # Learning rate
    decay_steps = int(opt.hyper.num_epochs_per_decay)
    lr = tf.train.exponential_decay(opt.hyper.learning_rate,
                                    global_step,
                                    decay_steps,
                                    opt.hyper.learning_rate_factor_per_decay,
                                    staircase=True)
    tf.summary.scalar('learning_rate', lr)
    tf.summary.scalar('weight_decay', opt.hyper.weight_decay)

    with tf.name_scope('accuracy'):
        probs = tf.nn.softmax(y)
        preds = tf.argmax(probs, 1)
        correct_prediction = tf.equal(preds, y_)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    ################################################################################################


    with tf.Session() as sess:

        flag_testable = False
        if not opt.skip_train:
            ################################################################################################
            # Set up Gradient Descent
            ################################################################################################
            all_var = tf.trainable_variables()
            train_step = tf.train.MomentumOptimizer(learning_rate=lr, momentum=opt.hyper.momentum).minimize(total_loss, var_list=all_var)
            inc_global_step = tf.assign_add(global_step, 1, name='increment')

            ################################################################################################
            # Set up checkpoints and data
            ################################################################################################
            saver = tf.train.Saver(max_to_keep=opt.max_to_keep_checkpoints)

            # Automatic restore model, or force train from scratch
            opt.restart = True # MAKE A TRAINING OPTION

            # Set up directories and checkpoints
            if not os.path.isfile(opt.log_dir_base + opt.name + '/models/checkpoint'):
                print("INIT")
                sess.run(tf.global_variables_initializer())
            elif opt.restart:
                print("RESTART")
                shutil.rmtree(opt.log_dir_base + opt.name + '/models/')
                shutil.rmtree(opt.log_dir_base + opt.name + '/train/')
                shutil.rmtree(opt.log_dir_base + opt.name + '/val/')
                sess.run(tf.global_variables_initializer())
            else:
                print("RESTORE")
                saver.restore(sess, tf.train.latest_checkpoint(opt.log_dir_base + opt.name + '/models/'))
                flag_testable = True

            # datasets
            # The `Iterator.string_handle()` method returns a tensor that can be evaluated
            # and used to feed the `handle` placeholder.
            training_handle = sess.run(train_iterator.string_handle())
            validation_handle = sess.run(val_iterator.string_handle())
            ################################################################################################

            ################################################################################################
            # RUN TRAIN
            ################################################################################################
            # Prepare summaries
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(opt.log_dir_base + opt.name + '/train', sess.graph)
            val_writer = tf.summary.FileWriter(opt.log_dir_base + opt.name + '/val')

            print("STARTING EPOCH = ", sess.run(global_step))
            ################################################################################################
            # Loop alternating between training and validation.
            ################################################################################################
            for iEpoch in range(int(sess.run(global_step)), opt.hyper.max_num_epochs):

                # Save metadata every epoch
                run_metadata = tf.RunMetadata()
                train_writer.add_run_metadata(run_metadata, 'epoch%03d' % iEpoch)
                saver.save(sess, opt.log_dir_base + opt.name + '/models/model', global_step=iEpoch)

                # Steps for doing one epoch
                for iStep in range(int(dataset.num_images_epoch/opt.hyper.batch_size)+1):

                    # Epoch counter
                    k = iStep*opt.hyper.batch_size + dataset.num_images_epoch*iEpoch

                    # Print accuray and summaries + train steps
                    if iStep == 0:
                        print("* epoch: " + str(float(k) / float(dataset.num_images_epoch)))
                        logits, labels, summ, acc_train, tl = sess.run(
                            [y, y_, merged, accuracy, total_loss],
                            feed_dict={handle: training_handle,
                                       dropout_rate: opt.hyper.drop_train
                                       })
                        train_writer.add_summary(summ, k)
                        print("train acc: " + str(acc_train))
                        print("train loss: " + str(tl))
                        sys.stdout.flush()

                        summ, acc_val, tl = sess.run([merged, accuracy, total_loss], feed_dict={handle: validation_handle,
                                                                                dropout_rate: opt.hyper.drop_test})
                        val_writer.add_summary(summ, k)
                        print("val acc: " + str(acc_val))
                        print("val loss: " + str(tl))
                        sys.stdout.flush()

                    else:
                        sess.run([train_step], feed_dict={handle: training_handle,
                                                          dropout_rate: opt.hyper.drop_train})

                sess.run([inc_global_step])
                print("----------------")
                sys.stdout.flush()
                ################################################################################################

            train_writer.close()
            val_writer.close()

        ################################################################################################
        # RUN TEST
        ################################################################################################

        else:
            sess.run(tf.global_variables_initializer())

        if flag_testable:
            print("MODEL WAS NOT TRAINED")

        import pickle
        acc = {}

        test_handle_full = sess.run(test_iterator_full.string_handle())
        val_handle_full = sess.run(val_iterator_full.string_handle())
        train_handle_full = sess.run(train_iterator_full.string_handle())

        def test_model(data_handle, data_iterator, name, acc):
            # Run one pass over a batch of the validation dataset.
            sess.run(data_iterator.initializer)
            acc_tmp = 0.0
            total = 0
            for num_iter in range(int(dataset.num_images_epoch/opt.hyper.batch_size)+1):
                acc_val, a, b, imm = sess.run(
                    [accuracy, preds, y_, image],
                    feed_dict={handle: data_handle, dropout_rate: opt.hyper.drop_test})

                acc_tmp += acc_val*len(a)
                total += len(a)

            acc[name] = acc_tmp / float(total)
            print("Total " + name + " = " + str(float(total)))
            print("Full " + name + " = " + str(acc[name]))
            sys.stdout.flush()
            return acc

        acc = test_model(train_handle_full, train_iterator_full, 'train', acc)
        acc = test_model(val_handle_full, val_iterator_full, 'val', acc)
        acc = test_model(test_handle_full, test_iterator_full, 'test', acc)

        if not os.path.exists(opt.log_dir_base + opt.name + '/results'):
            os.makedirs(opt.log_dir_base + opt.name + '/results')

        with open(opt.log_dir_base + opt.name + '/results/intra_dataset_accuracy.pkl', 'wb') as f:
            pickle.dump(acc, f)

        print(":)")



