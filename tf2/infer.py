import tensorflow as tf 
import numpy as np 
import sys, os, time, argparse, shutil, scipy, h5py, glob

parser = argparse.ArgumentParser(description='encode sinogram image.')
parser.add_argument('-gpus',  type=str, default="0", help='list of visiable GPUs')
parser.add_argument('-mdl', type=str, required=True, help='Experiment name')
parser.add_argument('-dsfn',type=str, default='../dataset/demo-dataset.h5', help='h5 dataset file')

args, unparsed = parser.parse_known_args()
if len(unparsed) > 0:
    print('Unrecognized argument(s): \n%s \nProgram exiting ... ... ' % '\n'.join(unparsed))
    exit(0)

if len(args.gpus) > 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disable printing INFO, WARNING, and ERROR

mdl = tf.keras.models.load_model(args.mdl, )

with h5py.File(args.dsfn, 'r') as h5fd:
    ns_img_test = h5fd['test_ns'][:]
    gt_img_test = h5fd['test_gt'][:]

if len(ns_img_test.shape) == 3:
    dn_img = mdl.predict(ns_img_test[:,:,:,np.newaxis]).squeeze()
elif len(ns_img_test.shape) == 4:
    dn_img = mdl.predict(ns_img_test).squeeze()
else:
    print("Model input must have N, H, W, C four dimension")

with h5py.File('dn.h5', 'w') as h5fd:
    h5fd.create_dataset("ns", data=ns_img_test, dtype=np.float32)
    h5fd.create_dataset("gt", data=gt_img_test, dtype=np.float32)
    h5fd.create_dataset("dn", data=dn_img, dtype=np.float32)
