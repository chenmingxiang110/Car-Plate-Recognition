import numpy as np
import cv2
import tensorflow as tf
from weapons.CTC_0a import ctc_recog_model

def sparseTuples2dense(sparseTensor):
    pred_dense = -np.ones(sparseTensor[2])
    for i in range(len(sparseTensor[0])):
        pred_dense[sparseTensor[0][i][0],sparseTensor[0][i][1]] = sparseTensor[1][i]
    return pred_dense

class recognizer:
    
    def __init__(self, model_path):
        tf.reset_default_graph()
        
        provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏",
                     "浙", "京", "闽", "赣", "鲁","豫", "鄂", "湘", "粤", "桂", "琼","川",
                     "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "_"]
        alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N',
                     'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '_']
        ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P',
               'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3',
               '4', '5', '6', '7', '8', '9', '_']
        self.labels_list = []
        for item in provinces+ads+alphabets:
            if item != "_" and item not in self.labels_list:
                self.labels_list.append(item)
        self.labels_list.append("_")
        
        self.model = ctc_recog_model(len(self.labels_list)+2)
        self.sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(self.sess, model_path)
    
    def predict(self, imgs):
        """
        imgs channels should RGB
        """
        x_shape = (300,150)
        xs = []
        for img in imgs:
            if np.max(img)>1:
                x = cv2.resize(img/255.1, x_shape)
            else:
                x = cv2.resize(x.astype(float), x_shape)
            xs.append(x)
        prediction = self.model.predict(self.sess, np.transpose(xs, axes = [0,2,1,3]),)
        prediction = sparseTuples2dense(prediction[0]).astype(int)
        results = []
        for p in prediction:
            results.append(''.join([self.labels_list[x] for x in p if x>=0 and x<len(self.labels_list)]))
        return results