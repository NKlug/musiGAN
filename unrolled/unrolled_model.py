import numpy as np

from bar import to_midi_pitch, save_as_midi
from network import generator, discriminator
import tensorflow as tf
from keras.optimizers import Adam
from tqdm import tqdm

from params import num_notes, magnitude_notes
from samples import create_train_dataset
from unrolled.unroll import extract_update_dict, graph_replace

feat_out = num_notes * magnitude_notes


class Model(object):

    def __init__(self):

        self.generator = lambda x: generator(x, feat_out)
        self.discriminator = lambda x: discriminator(x)
        self.global_step = tf.train.get_or_create_global_step()

        self.train_config()

        self.saver = tf.train.Saver(var_list=tf.trainable_variables())

    def train_config(self):

        self.gen_data = tf.placeholder(dtype=tf.float32, shape=[None, 2])
        self.real_data = tf.placeholder(dtype=tf.float32, shape=[None, num_notes, magnitude_notes])

        self.generated = self.generator(self.gen_data)

        self.real_score = self.discriminator(self.real_data)
        self.fake_score = self.discriminator(self.generated)

        trainable_variables = tf.trainable_variables()
        discriminator_variables = [var for var in trainable_variables if 'discriminator' in var.name]
        generator_variables = [var for var in trainable_variables if 'generator' in var.name]

        loss_fn = tf.nn.sigmoid_cross_entropy_with_logits

        self.generator_loss = loss_fn(logits=self.fake_score, labels=tf.ones_like(self.fake_score))
        self.discriminator_loss = (loss_fn(logits=self.real_score, labels=tf.ones_like(self.real_score))
                                   + loss_fn(logits=self.fake_score, labels=tf.zeros_like(self.fake_score))) / 2.0

        self.g_optimizer = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5)
        self.d_optimizer = Adam(lr=0.0002, beta_1=0.5)

        updates = self.d_optimizer.get_updates(discriminator_variables, [], self.discriminator_loss)
        self.d_train = tf.group(*updates)


        # Get dictionary mapping from variables to their update value after one optimization step
        update_dict = extract_update_dict(updates)
        cur_update_dict = update_dict
        for i in range(10 - 1):
            # Compute variable updates given the previous iteration's updated variable
            cur_update_dict = graph_replace(update_dict, cur_update_dict)
        # Final unrolled loss uses the parameters at the last time step
        unrolled_loss = graph_replace(self.discriminator_loss, cur_update_dict)

        self.g_train = self.g_optimizer.minimize(-unrolled_loss, var_list=generator_variables)


        # summary
        # self.g_loss_scalar = tf.summary.scalar('generator', self.generator_loss, family='loss')
        # self.d_loss_scalar = tf.summary.scalar('discriminator', self.discriminator_loss, family='loss')

    def restore(self, sess, checkpoint):
        print("Restoring ", checkpoint)
        self.saver.restore(sess, checkpoint)

    def train(self, sess):
        num_steps = 1500

        tf.global_variables_initializer().run()
        current_step = sess.run(self.global_step)
        print('Global step: {}'.format(current_step))

        summary = tf.summary.merge_all(scope='loss')
        writer = tf.summary.FileWriter('~/log')

        pbar = tqdm(range(num_steps))


        real_data = create_train_dataset()

        for x in pbar:

            gen_data = np.random.rand(4, 2) * 2 - 1
            i = (x * 4) % (len(real_data) - 4)
            t_data = real_data[i:i + 4]

            s, d_loss, g_loss, generated, *_ = sess.run(
                [summary, self.discriminator_loss, self.generator_loss, self.generated, self.g_train, self.d_train],
                feed_dict={
                    self.gen_data: gen_data,
                    self.real_data: t_data
                })
            print(np.argmax(generated, axis=-1))

            # writer.add_summary(s, x)
            pbar.set_description("d_loss: {:10.2f}, g_loss: {:10.2f}".format(d_loss, g_loss))
            print("d_loss: {:10.2f}, g_loss: {:10.2f}".format(d_loss, g_loss))
            # if x % 1000 == 0:
            #     self.saver.save(sess=sess, save_path='~/ckpt/model', global_step=self.global_step)

        self.saver.save(sess=sess, save_path='/home/nklug/ckpt/model.ckpt')

    def infer(self, sess, ran, checkpoint):
        ran = np.asarray(ran)
        if len(ran.shape) == 1:
            ran = ran[np.newaxis, :]

        self.restore(sess, checkpoint)
        generated = sess.run(self.generated, feed_dict={self.gen_data: ran})
        return np.argmax(generated, axis=-1)


if __name__ == '__main__':
    with tf.Session() as sess:
        model = Model()
        model.train(sess)

        np.random.seed(0)
        r = np.random.rand(10, 2) * 2 - 1
        print(r)
        print()

        generated = model.infer(sess, r, '/home/nklug/ckpt/model.ckpt')
        print(generated)
        generated = to_midi_pitch(generated[0])
        save_as_midi(generated, 'tmp.midi')
