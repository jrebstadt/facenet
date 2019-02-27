import tensorflow as tf
import numpy as np
from src import facenet
from src.align import detect_face
from scipy import misc
import os

MODEL = os.path.abspath(os.path.join(os.path.join(os.getcwd(), os.pardir), "saved_model/20180402-114759"))
GPU_MEMORY_FRACTION = 1.0
MARGIN = 44
IMAGE_SIZE = 160


def compute_embedding(image):

    image = load_and_align_image(image, IMAGE_SIZE, MARGIN, GPU_MEMORY_FRACTION)
    image = image[None, ...]
    with tf.Graph().as_default():

        with tf.Session() as sess:

            # Load the model
            facenet.load_model(MODEL)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # Run forward pass to calculate embeddings
            feed_dict = { images_placeholder: image, phase_train_placeholder :False }
            embedding = sess.run(embeddings, feed_dict=feed_dict)

    return embedding



def load_and_align_image(image, image_size, margin, gpu_memory_fraction):

    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor

    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

    img_size = np.asarray(image.shape)[0:2]
    bounding_boxes, _ = detect_face.detect_face(image, minsize, pnet, rnet, onet, threshold, factor)
    if len(bounding_boxes) < 1:
        print("can't detect face, remove ", image)
    det = np.squeeze(bounding_boxes[0 ,0:4])
    bb = np.zeros(4, dtype=np.int32)
    bb[0] = np.maximum(det[0 ] - margin /2, 0)
    bb[1] = np.maximum(det[1 ] - margin /2, 0)
    bb[2] = np.minimum(det[2 ] + margin /2, img_size[1])
    bb[3] = np.minimum(det[3 ] + margin /2, img_size[0])
    cropped = image[bb[1]:bb[3] ,bb[0]:bb[2] ,:]
    aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
    prewhitened = facenet.prewhiten(aligned)

    return prewhitened
