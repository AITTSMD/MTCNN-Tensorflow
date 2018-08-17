import tensorflow as tf
import numpy as np

def cal_recall(cls_pred,label):
    predictions = tf.argmax(cls_pred,axis = 1)

