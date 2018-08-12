# python train_model.py model={iphone,sony,blackberry} dped_dir=dped vgg_dir=vgg_pretrained/imagenet-vgg-verydeep-19.mat

import tensorflow as tf
from scipy import misc
import os
import numpy as np
import sys
from experiments import config
from data.load_dataset import load_test_data, load_batch
from metrics import MultiScaleSSIM
from metrics import PSNR
from net import EDSR
from loss import color_loss,content_loss,variation_loss,texture_loss


# processing command arguments

phone, batch_size, train_size, test_size, learning_rate, num_train_iters, \
w_content, w_color, w_texture, w_tv, \
dped_dir, vgg_dir, eval_step, summary_step,\
PATCH_WIDTH, PATCH_HEIGHT, PATCH_SIZE, \
models_dir, result_dir, checkpoint_dir = config.process_command_args(sys.argv)

np.random.seed(0)

# loading training and test data

print("Loading test data...")
test_data, test_answ = load_test_data(dped_dir, test_size)
print test_data.shape
print("Test data was loaded\n")

print("Loading training data...")
train_data, train_answ = load_batch(dped_dir, train_size)
print train_data.shape
print("Training data was loaded\n")

TEST_SIZE = test_data.shape[0]
num_test_batches = int(test_data.shape[0]/batch_size)

# defining system architecture

with tf.Graph().as_default(), tf.Session() as sess:
    
    # placeholders for training data

    phone_ = tf.placeholder(tf.float32, [None, PATCH_SIZE])
    phone_image = tf.reshape(phone_, [-1, PATCH_HEIGHT, PATCH_WIDTH, 3])

    dslr_ = tf.placeholder(tf.float32, [None, PATCH_SIZE])
    dslr_image = tf.reshape(dslr_, [-1, PATCH_HEIGHT, PATCH_WIDTH, 3])

    # adv_ = tf.placeholder(tf.float32, [None, 1])

    enhanced = EDSR(phone_image)
    print enhanced.shape

    #loss introduce
    # loss_texture, discim_accuracy = texture_loss(enhanced,dslr_image,PATCH_WIDTH,PATCH_HEIGHT,adv_)
    # loss_discrim = -loss_texture
    # loss_content = content_loss(vgg_dir,enhanced,dslr_image,batch_size)
    # loss_color = color_loss(enhanced, dslr_image, batch_size)
    # loss_tv = variation_loss(enhanced,PATCH_WIDTH,PATCH_HEIGHT,batch_size)

    # loss_generator = w_content * loss_content + w_texture * loss_texture + w_color * loss_color + w_tv * loss_tv
    loss_generator = tf.losses.absolute_difference(labels=dslr_image, predictions=enhanced)
    loss_psnr = PSNR(enhanced,dslr_,PATCH_SIZE,batch_size)
    loss_ssim = MultiScaleSSIM(enhanced, dslr_image)

    # optimize parameters of image enhancement (generator) and discriminator networks
    generator_vars = [v for v in tf.global_variables() if v.name.startswith("generator")]
    # discriminator_vars = [v for v in tf.global_variables() if v.name.startswith("discriminator")]

    train_step_gen = tf.train.AdamOptimizer(learning_rate).minimize(loss_generator, var_list=generator_vars)
    # train_step_disc = tf.train.AdamOptimizer(learning_rate).minimize(loss_discrim, var_list=discriminator_vars)

    saver = tf.train.Saver(var_list=generator_vars, max_to_keep=100)

    print('Initializing variables')
    sess.run(tf.global_variables_initializer())

    print('Training network')

    train_loss_gen = 0.0
    # train_acc_discrim = 0.0

    # all_zeros = np.reshape(np.zeros((batch_size, 1)), [batch_size, 1])
    index = np.random.randint(0, TEST_SIZE, 5)
    test_crops = test_data[index, :]
    test_crops_ans = test_answ[index, :]

    logs = open('../models/' + phone + '.txt', "w+")
    logs.close()


    #summary ,add the scalar you want to see

    tf.summary.scalar('loss_generator', loss_generator),
    # tf.summary.scalar('loss_content', loss_content),
    # tf.summary.scalar('loss_color', loss_color),
    # tf.summary.scalar('loss_texture', loss_texture),
    # tf.summary.scalar('loss_tv', loss_tv),
    # tf.summary.scalar('discim_accuracy', discim_accuracy),
    tf.summary.scalar('psnr', loss_psnr),
    tf.summary.scalar('ssim', loss_ssim),
    tf.summary.scalar('learning_rate', learning_rate),
    merge_summary = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(models_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(models_dir + '/test', sess.graph)
    tf.global_variables_initializer().run()

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print('loading checkpoint')
        saver.restore(sess, ckpt.model_checkpoint_path)



    for i in range(num_train_iters):

        # train generator

        idx_train = np.random.randint(0, train_size, batch_size)

        phone_images = train_data[idx_train]
        dslr_images = train_answ[idx_train]

        [loss_temp, temp] = sess.run([loss_generator, train_step_gen],
                                        feed_dict={phone_: phone_images, dslr_: dslr_images})
        train_loss_gen += loss_temp / eval_step

        # train discriminator

        # idx_train = np.random.randint(0, train_size, batch_size)

        # generate image swaps (dslr or enhanced) for discriminator
        # swaps = np.reshape(np.random.randint(0, 2, batch_size), [batch_size, 1])

        # phone_images = train_data[idx_train]
        # dslr_images = train_answ[idx_train]

        # [accuracy_temp, temp] = sess.run([discim_accuracy, train_step_disc],
        #                                 feed_dict={phone_: phone_images, dslr_: dslr_images, adv_: swaps})
        # train_acc_discrim += accuracy_temp / eval_step

        if i % summary_step == 0:
            #summary intervals
            train_summary = sess.run(merge_summary, feed_dict={phone_: phone_images, dslr_: dslr_images})
            train_writer.add_summary(train_summary, i)

        if i % eval_step == 0:
            # test generator and discriminator CNNs
            test_losses_gen = np.zeros((1, 3))
            # test_accuracy_disc = 0.0

            for j in range(num_test_batches):

                be = j * batch_size
                en = (j+1) * batch_size

                # swaps = np.reshape(np.random.randint(0, 2, batch_size), [batch_size, 1])

                phone_images = test_data[be:en]
                dslr_images = test_answ[be:en]

                [enhanced_crops, losses] = sess.run([enhanced, [loss_generator, loss_psnr, loss_ssim]], \
                                feed_dict={phone_: phone_images, dslr_: dslr_images})

                test_losses_gen += np.asarray(losses) / num_test_batches
                # test_accuracy_disc += accuracy_disc / num_test_batches

                # loss_ssim += MultiScaleSSIM(np.reshape(dslr_images * 255, [batch_size, PATCH_HEIGHT, PATCH_WIDTH, 3]),
                #                                     enhanced_crops * 255) / num_test_batches

            # logs_disc = "step %d, %s | discriminator accuracy | train: %.4g, test: %.4g" % \
            #       (i, phone, train_acc_discrim, test_accuracy_disc)

            logs_gen = "generator losses | train: %.4g, test: %.4g | psnr: %.4g, ssim: %.4g\n" % \
                  (train_loss_gen, test_losses_gen[0][0], test_losses_gen[0][1], test_losses_gen[0][2])

            # print(logs_disc)
            print(logs_gen)

            test_summary = sess.run(merge_summary, feed_dict={phone_: phone_images, dslr_: dslr_images})
            test_writer.add_summary(test_summary, i)

            # save the results to log file

            logs = open(models_dir + phone + '.txt', "a")
            # logs.write(logs_disc)
            # logs.write('\n')
            logs.write(logs_gen)
            logs.write('\n')
            logs.close()

            # save visual results for several test image crops

            enhanced_crops = sess.run(enhanced, feed_dict={phone_: test_crops, dslr_: dslr_images})

            idx = 0
            for crop in enhanced_crops:
                before_after = np.hstack((np.reshape(test_crops[idx], [PATCH_HEIGHT, PATCH_WIDTH, 3]), crop, np.reshape(test_crops_ans[idx], [PATCH_HEIGHT, PATCH_WIDTH, 3])))
                before_after = before_after * 255 + 0.5
                before_after = np.clip(before_after, 0.0, 255.0)
                before_after = before_after.astype(np.uint8)
                misc.imsave(result_dir + str(phone)+ "_" + str(idx) + '_iteration_' + str(i) + '.jpg', before_after)
                idx += 1

            train_loss_gen = 0.0
            # train_acc_discrim = 0.0

            # save the model that corresponds to the current iteration

            saver.save(sess, models_dir + str(phone) + '_iteration_' + str(i) + '.ckpt', write_meta_graph=False)

            # reload a different batch of training data

            del train_data
            del train_answ
            train_data, train_answ = load_batch(dped_dir, train_size)

        if KeyboardInterrupt:
            saver.save(sess, models_dir + str(phone) + '_iteration_' + 'on' + '.ckpt', write_meta_graph=False)
