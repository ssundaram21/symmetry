import os.path
import shutil
import sys
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "0"
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
    ################################################################################################

    if os.path.isfile(opt.log_dir_base + opt.name + '/results/intra_dataset_accuracy.pkl'):
        print(":)")
        quit()

    #print(opt.hyper.complex_crossing)
    print(opt.hyper.init_factor)
    print(opt.hyper.max_num_epochs)
    print(opt.hyper.learning_rate)
    print(opt.hyper.alpha)
    print(opt.hyper.batch_size)

    #tf.logging.set_verbosity(tf.logging.INFO)

    ################################################################################################
    # Define training and validation datasets through Dataset API
    ################################################################################################

    # Initialize dataset and creates TF records if they do not exist

    if opt.dataset.dataset_name == 'insideness':
        from data import insideness_data
        dataset = insideness_data.InsidenessDataset(opt)
    else:
        print("Error: no valid dataset specified")

    # Repeatable datasets for training
    train_dataset = dataset.create_dataset(augmentation=False, standarization=False, set_name='train', repeat=True)
    val_dataset = dataset.create_dataset(augmentation=False, standarization=False, set_name='val', repeat=True)

    # No repeatable dataset for testing
    train_dataset_full = dataset.create_dataset(augmentation=False, standarization=False, set_name='train', repeat=True)
    val_dataset_full = dataset.create_dataset(augmentation=False, standarization=False, set_name='val', repeat=True)
    test_dataset_full = dataset.create_dataset(augmentation=False, standarization=False, set_name='test', repeat=True)

    # Hadles to switch datasets
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

    # Call DNN
    dropout_rate = tf.placeholder(tf.float32)
    to_call = getattr(nets, opt.dnn.name)
    y, parameters, activations = to_call(image, opt, dropout_rate, len(dataset.list_labels)*dataset.num_outputs)

    # Loss function
    with tf.name_scope('loss'):
        weights_norm = tf.reduce_sum(
            input_tensor=opt.hyper.weight_decay * tf.stack(
                [tf.nn.l2_loss(i) for i in parameters]
            ),
            name='weights_norm')
        tf.summary.scalar('weight_decay', weights_norm)

        flat_y_ = tf.reshape(tensor=y_, shape=[-1, opt.dataset.image_size ** 2])
        flat_image = tf.reshape(tensor=tf.cast(image, tf.int64), shape=[-1, opt.dataset.image_size ** 2])
        im = tf.cast((flat_image), tf.float32)
        cl = tf.cast(flat_y_, tf.float32)

        flag_loss_per_step = False
        if hasattr(opt.dnn, 'train_per_step'):
            if opt.dnn.train_per_step:
                flag_loss_per_step = True
                print("TRAIN PER STEP")
                sys.stdout.flush()
                cross_list = []
                for yy in y:
                    flat_y = tf.reshape(tensor=yy, shape=[-1, opt.dataset.image_size ** 2, len(dataset.list_labels)])
                    cross_tmp = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=flat_y_, logits=flat_y)
                    cross_list.append(tf.reduce_mean( \
                        (1 - opt.hyper.alpha) * tf.reduce_sum(((1 - im) * cl) * cross_tmp, 1) / tf.reduce_sum(
                            (1 - im) * cl, 1) + \
                        (opt.hyper.alpha) * tf.reduce_sum(((1 - im) * (1 - cl)) * cross_tmp, 1) / tf.reduce_sum(
                            (1 - im) * (1 - cl), 1)))

                cross_entropy_sum = tf.add_n(cross_list)

        #If loss is for the last state
        if not flag_loss_per_step:
            print("TRAIN AT END")
            sys.stdout.flush()
            flat_y = tf.reshape(tensor=y, shape=[-1, opt.dataset.image_size ** 2, len(dataset.list_labels)])
            cross = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=flat_y_, logits=flat_y)
            cross_entropy_sum = tf.reduce_mean( \
                (1 - opt.hyper.alpha) * tf.reduce_sum(((1 - im) * cl) * cross, 1) / tf.reduce_sum((1 - im) * cl, 1) + \
                (opt.hyper.alpha) * tf.reduce_sum(((1 - im) * (1 - cl)) * cross, 1) / tf.reduce_sum((1 - im) * (1 - cl),
                                                                                                    1))

        tf.summary.scalar('cross_entropy', cross_entropy_sum)

        total_loss = weights_norm + cross_entropy_sum
        tf.summary.scalar('total_loss', total_loss)

    global_step = tf.Variable(0, name='global_step', trainable=False)
    ################################################################################################


    ################################################################################################
    # Set up Training
    ################################################################################################

    # Learning rate
    num_batches_per_epoch = dataset.num_images_epoch / opt.hyper.batch_size
    decay_steps = int(opt.hyper.num_epochs_per_decay)
    lr = tf.train.exponential_decay(opt.hyper.learning_rate,
                                    global_step,
                                    decay_steps,
                                    opt.hyper.learning_rate_factor_per_decay,
                                    staircase=True)
    tf.summary.scalar('learning_rate', lr)
    tf.summary.scalar('weight_decay', opt.hyper.weight_decay)

    # Accuracy
    with tf.name_scope('accuracy'):
        flat_output = tf.argmax(flat_y, 2)
        correct_prediction = tf.equal(flat_output * (1 - flat_image), flat_y_ * (1 - flat_image))
        correct_prediction = tf.cast(correct_prediction, tf.float32)

        error_images = tf.reduce_min(correct_prediction, 1)
        accuracy = tf.reduce_mean(error_images)

        accuracy_loose = tf.reduce_mean(
            0.5*tf.reduce_sum(((1 - im) * cl) * correct_prediction, 1) / tf.reduce_sum((1 - im) * cl, 1) + \
            0.5*tf.reduce_sum(((1 - im) * (1 - cl)) * correct_prediction, 1) / tf.reduce_sum((1 - im) * (1 - cl), 1))

        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('accuracy_loose', accuracy_loose)

    if opt.extense_summary:
        tf.summary.image('input', tf.expand_dims(
            tf.reshape(tf.cast(flat_image, tf.float32), [-1, opt.dataset.image_size, opt.dataset.image_size]), 3))
            #image, 3))
        tf.summary.image('output', tf.expand_dims(
            tf.reshape(tf.cast(flat_output, tf.float32), [-1, opt.dataset.image_size, opt.dataset.image_size]), 3))
        tf.summary.image('output1', tf.expand_dims(
            tf.reshape(tf.cast(flat_y[:,:,0], tf.float32), [-1, opt.dataset.image_size, opt.dataset.image_size]), 3))
        tf.summary.image('output2', tf.expand_dims(
            tf.reshape(tf.cast(flat_y[:,:,1], tf.float32), [-1, opt.dataset.image_size, opt.dataset.image_size]), 3))
        tf.summary.image('gt', tf.expand_dims(
            tf.reshape(tf.cast(flat_y_, tf.float32), [-1, opt.dataset.image_size, opt.dataset.image_size]), 3))
        tf.summary.image('correctness', tf.expand_dims(
            tf.reshape(tf.cast(correct_prediction, tf.float32), [-1, opt.dataset.image_size, opt.dataset.image_size]), 3))

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

            raw_grads = tf.gradients(total_loss, all_var)
            grads = list(zip(raw_grads, tf.trainable_variables()))

            for g, v in grads:
                summary.gradient_summaries(g, v, opt)
            ################################################################################################


            ################################################################################################
            # Set up checkpoints and data
            ################################################################################################

            saver = tf.train.Saver(max_to_keep=opt.max_to_keep_checkpoints)

            # Automatic restore model, or force train from scratch


            # Set up directories and checkpoints
            if not os.path.isfile(opt.log_dir_base + opt.name + '/models/checkpoint'):
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
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summ = sess.run([merged], feed_dict={handle: training_handle, dropout_rate: opt.hyper.drop_train},
                                   options=run_options, run_metadata=run_metadata)
                train_writer.add_run_metadata(run_metadata, 'epoch%03d' % iEpoch)
                saver.save(sess, opt.log_dir_base + opt.name + '/models/model', global_step=iEpoch)

                # Steps for doing one epoch
                for iStep in range(int(dataset.num_images_epoch/opt.hyper.batch_size)+1):

                    # Epoch counter
                    k = iStep*opt.hyper.batch_size + dataset.num_images_epoch*iEpoch

                    # Print accuray and summaries + train steps
                    if iStep == 0:
                        # !train_step
                        print("* epoch: " + str(float(k) / float(dataset.num_images_epoch)))
                        summ, acc_train, acc_loo, tl = sess.run([merged, accuracy, accuracy_loose, total_loss],
                                                        feed_dict={handle: training_handle,
                                                                   dropout_rate: opt.hyper.drop_train})
                        train_writer.add_summary(summ, k)
                        print("train acc: " + str(acc_train))
                        print("train acc loose: " + str(acc_loo))
                        print("train loss: " + str(tl))
                        sys.stdout.flush()

                        summ, acc_val, acc_loo, tl = sess.run([merged, accuracy, accuracy_loose,  total_loss], feed_dict={handle: validation_handle,
                                                                                dropout_rate: opt.hyper.drop_test})
                        val_writer.add_summary(summ, k)
                        print("val acc: " + str(acc_val))
                        print("val acc loose: " + str(acc_loo))
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
            acc_tmp_loo = 0.0
            total = 0
            for num_iter in range(int(dataset.num_images_epoch/opt.hyper.batch_size)+1):
                acc_val, acc_loo, a, b, err, imm = sess.run(
                    [accuracy, accuracy_loose, flat_output, y_, error_images, image],
                    feed_dict={handle: data_handle, dropout_rate: opt.hyper.drop_test})

                ''' 
                if 0 in err:
                    #import matplotlib as mpl
                    #mpl.use('Agg')
                    import matplotlib.pyplot as plt
                    mm = (err == 0)
                    from PIL import Image;
                    bb = np.reshape(imm[mm, :, :].astype(np.uint8),[32,32])
                    imga = Image.fromarray(128 * bb);
                    imga.save('testrgb2.png')
                    aa = np.reshape(a[mm, :].astype(np.uint8), [32, 32])
                    imga = Image.fromarray(128 * aa);
                    imga.save('testrgb1.png')
                '''

                acc_tmp_loo += acc_loo * len(a)
                acc_tmp += acc_val*len(a)
                total += len(a)

            acc[name] = acc_tmp / float(total)
            acc[name + 'loose'] = acc_tmp_loo / float(total)
            print("Total " + name + " = " + str(float(total)))
            print("Full " + name + " = " + str(acc[name]))
            print("Full " + name + " loose = " + str(acc[name + 'loose']))
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



