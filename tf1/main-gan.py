import tensorflow as tf 
tf.enable_eager_execution()
import numpy as np 
from util import save2img, str2bool
import sys, os, time, argparse, shutil, scipy, h5py, glob
from models import tomogan_disc as make_discriminator_model  # import a disc model
from models import unet as make_generator_model           # import a generator model
from data import bkgdGen, gen_train_batch_bg, get1batch4test

tf.logging.set_verbosity(tf.logging.ERROR)

parser = argparse.ArgumentParser(description='TomoGAN, for noise/artifact removal')
parser.add_argument('-gpus',  type=str, default="0", help='list of visiable GPUs')
parser.add_argument('-expName', type=str, default='debug', help='Experiment name')
parser.add_argument('-lmse',  type=float, default=0.5, help='lambda mse')
parser.add_argument('-ladv',  type=float, default=20, help='lambda adv')
parser.add_argument('-lperc', type=float, default=2, help='lambda perceptual')
parser.add_argument('-lunet', type=int, default=3, help='Unet layers')
parser.add_argument('-depth', type=int, default=1, help='input depth (use for 3D CT image only)')
parser.add_argument('-psz',   type=int, default=256, help='cropping patch size')
parser.add_argument('-mbsz',  type=int, default=16, help='mini-batch size')
parser.add_argument('-itg',   type=int, default=1, help='iterations for G')
parser.add_argument('-itd',   type=int, default=2, help='iterations for D')
parser.add_argument('-maxiter', type=int, default=8000, help='maximum iterations')
parser.add_argument('-dsfn',  type=str, required=True, help='h5 dataset file')
parser.add_argument('-print', type=str2bool, default=False, help='1: print to terminal; 0: redirect to file')

args, unparsed = parser.parse_known_args()
if len(unparsed) > 0:
    print('Unrecognized argument(s): \n%s \nProgram exiting ... ... ' % '\n'.join(unparsed))
    exit(0)

if len(args.gpus) > 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disable printing INFO, WARNING, and ERROR

itr_out_dir = args.expName + '-itrOut'
if os.path.isdir(itr_out_dir): 
    shutil.rmtree(itr_out_dir)
os.mkdir(itr_out_dir) # to save temp output

if args.print == 0:
    sys.stdout = open('%s/%s' % (itr_out_dir, 'iter-prints.log'), 'w') 

# build minibatch data generator with prefetch
mb_data_iter = bkgdGen(data_generator=gen_train_batch_bg(
                                      dsfn=args.dsfn, mb_size=args.mbsz, \
                                      in_depth=args.depth, img_size=args.psz), \
                       max_prefetch=args.mbsz*4)

generator = make_generator_model(input_shape=(None, None, args.depth), nlayers=args.lunet ) 
discriminator = make_discriminator_model(input_shape=(args.psz, args.psz, 1))

feature_extractor_vgg = tf.keras.applications.VGG19(\
                        weights='../vgg19_weights_notop.h5', \
                        include_top=False)

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def adversarial_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

gen_optimizer  = tf.train.AdamOptimizer(1e-4)
disc_optimizer = tf.train.AdamOptimizer(1e-4)

ckpt = tf.train.Checkpoint(generator_optimizer=gen_optimizer,
                            discriminator_optimizer=disc_optimizer,
                            generator=generator,
                            discriminator=discriminator)

for epoch in range(args.maxiter+1):
    time_git_st = time.time()
    for _ge in range(args.itg):
        X_mb, y_mb = mb_data_iter.next() # with prefetch
        with tf.GradientTape() as gen_tape:
            gen_tape.watch(generator.trainable_variables)

            gen_imgs = generator(X_mb, training=True)
            disc_fake_o = discriminator(gen_imgs, training=False)

            loss_mse = tf.losses.mean_squared_error(gen_imgs, y_mb)
            loss_adv = adversarial_loss(disc_fake_o)

            _img_gt = tf.keras.applications.vgg19.preprocess_input(np.concatenate([y_mb, y_mb, y_mb], 3))
            vggf_gt = feature_extractor_vgg.predict(_img_gt)

            _img_dn = tf.keras.applications.vgg19.preprocess_input(np.concatenate([gen_imgs, gen_imgs, gen_imgs], 3))
            vggf_gen = feature_extractor_vgg.predict(_img_dn)

            perc_loss= tf.losses.mean_squared_error(vggf_gt.reshape(-1), vggf_gen.reshape(-1))

            gen_loss = args.ladv * loss_adv + args.lmse * loss_mse + args.lperc * perc_loss

            gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
            gen_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))

    itr_prints_gen = '[Info] Epoch: %05d, gloss: %.2f (mse%.3f, adv%.3f, perc:%.3f), gen_elapse: %.2fs/itr' % (\
                     epoch, gen_loss, loss_mse*args.lmse, loss_adv*args.ladv, perc_loss*args.lperc, \
                     (time.time() - time_git_st)/args.itg, )
    time_dit_st = time.time()

    for _de in range(args.itd):
        X_mb, y_mb = mb_data_iter.next() # with prefetch 
        with tf.GradientTape() as disc_tape:
            disc_tape.watch(discriminator.trainable_variables)

            gen_imgs = generator(X_mb, training=False)

            disc_real_o = discriminator(y_mb, training=True)
            disc_fake_o = discriminator(gen_imgs, training=True)

            disc_loss = discriminator_loss(disc_real_o, disc_fake_o)

            disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
            disc_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))

    print('%s; dloss: %.2f (r%.3f, f%.3f), disc_elapse: %.2fs/itr, gan_elapse: %.2fs/itr' % (itr_prints_gen,\
          disc_loss, disc_real_o.numpy().mean(), disc_fake_o.numpy().mean(), \
          (time.time() - time_dit_st)/args.itd, time.time()-time_git_st))

    if epoch % (500//args.itg) == 0:
        X222, y222 = get1batch4test(dsfn=args.dsfn, in_depth=args.depth)
        pred_img = generator.predict(X222[:1])

        save2img(pred_img[0,:,:,0], '%s/it%05d.png' % (itr_out_dir, epoch))
        if epoch == 0: 
            save2img(y222[0,:,:,0], '%s/gt.png' % (itr_out_dir))
            save2img(X222[0,:,:,args.depth//2], '%s/ns.png' % (itr_out_dir))

        generator.save("%s/%s-it%05d.h5" % (itr_out_dir, args.expName, epoch), \
                       include_optimizer=False)
        # discriminator.save("%s/disc-it%05d.h5" % (itr_out_dir, epoch), \
        #                include_optimizer=False)

    sys.stdout.flush()

