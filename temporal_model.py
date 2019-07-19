import numpy as np

from bar import to_midi_pitch, save_as_midi, dataset_from_midi, temporal_dataset_from_midi, save_as_temporal_midi, \
    round_to
from clipping import adaptive_clipping_fn
from losses import softmax_x_entropy
from network import generator, discriminator, temporal_generator
import tensorflow as tf
from tqdm import tqdm
import os

from params import num_notes, magnitude_notes, rand_dim, temp_normalize
from samples import create_train_dataset

feat_out = num_notes * (magnitude_notes + 1)


def label_smoothing(data):
    return data * 0.6 + 0.2


class Model(object):

    def __init__(self):

        self.generator = lambda x: temporal_generator(x, feat_out)
        self.discriminator = lambda x: discriminator(x)
        self.global_step = tf.train.get_or_create_global_step()

        self.train_config()

        self.saver = tf.train.Saver(var_list=tf.trainable_variables())

    def train_config(self):

        self.gen_data = tf.placeholder(dtype=tf.float32, shape=[None, rand_dim])
        self.real_data = tf.placeholder(dtype=tf.float32, shape=[None, num_notes, magnitude_notes + 1])

        self.generated = self.generator(self.gen_data)

        self.real_score = self.discriminator(self.real_data)
        self.fake_score = self.discriminator(self.generated)

        trainable_variables = tf.trainable_variables()
        discriminator_variables = [var for var in trainable_variables if 'discriminator' in var.name]
        generator_variables = [var for var in trainable_variables if 'generator' in var.name]

        activation = softmax_x_entropy

        self.generator_loss = tf.reduce_mean(activation(logits=self.fake_score, labels=1))
        self.discriminator_loss = tf.reduce_mean(
            [activation(logits=self.real_score, labels=1),
             activation(logits=self.fake_score, labels=0)])

        self.g_optimizer = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5)
        self.d_optimizer = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5)

        clip_fn = adaptive_clipping_fn(global_step=self.global_step, static_max_norm=1.0)

        g_grads_vars = self.g_optimizer.compute_gradients(self.generator_loss, var_list=generator_variables)
        self.g_train = self.g_optimizer.apply_gradients(clip_fn(g_grads_vars), global_step=self.global_step)

        d_grads_vars = self.d_optimizer.compute_gradients(self.discriminator_loss,
                                                          var_list=discriminator_variables)
        self.d_train = self.d_optimizer.apply_gradients(clip_fn(d_grads_vars),
                                                        global_step=self.global_step)

        # summary
        self.g_loss_scalar = tf.summary.scalar('generator', self.generator_loss, family='loss')
        self.d_loss_scalar = tf.summary.scalar('discriminator', self.discriminator_loss, family='loss')

    def restore(self, sess, checkpoint):
        print("Restoring ", checkpoint)
        self.saver.restore(sess, checkpoint)

    def train(self, sess, checkpoint):
        num_steps = 10000
        batch_size = 100

        tf.global_variables_initializer().run()
        current_step = sess.run(self.global_step)
        print('Global step: {}'.format(current_step))

        summary = tf.summary.merge_all(scope='loss')
        writer = tf.summary.FileWriter('~/log')

        pbar = tqdm(range(num_steps))

        # real_data = create_train_dataset()
        real_data = temporal_dataset_from_midi('/home/nklug/mgan/zauberflute.mid')

        # real_data = label_smoothing(real_data)

        for x in pbar:
            gen_data = np.random.normal(size=(batch_size, rand_dim))
            i = (x * batch_size) % (len(real_data) - batch_size)
            t_data = real_data[i:i + batch_size]

            s, d_loss, g_loss, generated, *_ = sess.run(
                [summary, self.discriminator_loss, self.generator_loss, self.generated, self.g_train, self.d_train],
                feed_dict={
                    self.gen_data: gen_data,
                    self.real_data: t_data
                })
            # print(np.argmax(generated, axis=-1)[0])

            # writer.add_summary(s, x)
            pbar.set_description("d_loss: {:10.2f}, g_loss: {:10.2f}".format(d_loss, g_loss))
            # print("d_loss: {:10.2f}, g_loss: {:10.2f}".format(d_loss, g_loss))
            # if x % 1000 == 0:
            #     self.saver.save(sess=sess, save_path='~/ckpt/model', global_step=self.global_step)

        self.saver.save(sess=sess, save_path=checkpoint)

    def infer(self, sess, ran, checkpoint):
        ran = np.asarray(ran)
        if len(ran.shape) == 1:
            ran = ran[np.newaxis, :]

        self.restore(sess, checkpoint)
        generated = sess.run(self.generated, feed_dict={self.gen_data: ran})
        return np.round(np.concatenate(
            [np.argmax(generated[:, :, :-1], axis=-1)[:, :, np.newaxis], np.abs(generated[:, :, -1:] * temp_normalize)],
            axis=-1), decimals=1)


if __name__ == '__main__':
    name = 'temporal'
    checkpoint_dir = '/home/nklug/ckpt/{}'.format(name)
    checkpoint = os.path.join(checkpoint_dir, 'model{}.ckpt'.format(num_notes))
    os.makedirs(checkpoint_dir, exist_ok=True)
    with tf.Session() as sess:
        model = Model()
        # model.train(sess, checkpoint)

        np.random.seed(0)
        r = np.random.normal(size=(10, rand_dim))
        print(r)
        print()

        generated = model.infer(sess, r, checkpoint)
        print(generated)
        out_dir = 'results/{}'.format(name)
        os.makedirs(out_dir, exist_ok=True)
        for i, gen in enumerate(generated):
            generated = to_midi_pitch(gen[:, :1])
            generated = np.concatenate([generated, round_to(gen[:, 1:], 32)], axis=-1)
            save_as_temporal_midi(generated, os.path.join(out_dir, 'out_{}_{}.mid'.format(num_notes, i)),
                                  ticks_per_beat=120)
