import numpy as np
import cv2
import tensorflow as tf
from weapons.Se_0a import seg_model

def seg_to_vertices(img, use_dilated = False):
    if use_dilated:
        # dilate thresholded image - merges top/bottom 
        kernel = np.ones((3,3))
        dilated = cv2.dilate(img, kernel, iterations=3)
        current_img = dilated
    else:
        current_img = img

    # find contours
    contours, hierarchy = cv2.findContours(current_img.astype(np.uint8),
                                           cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if len(contours)<=0 or len(contours[0])<4:
        return []
    
    # simplify contours
    used_index = set([])
    index = 0.01
    step = 2
    while(len(used_index)<10):
        epsilon = index*cv2.arcLength(contours[0],True)
        approx = cv2.approxPolyDP(contours[0],epsilon,True)
        if len(approx)==4:
            break
        elif len(approx)>4:
            if (index*step) in used_index:
                step = 1+(step-1)/2
                used_index.add(index*step)
                index*=step
            else:
                used_index.add(index*step)
                index*=step
        else:
            if (index/step) in used_index:
                step = 1+(step-1)/2
                used_index.add(index/step)
                index/=step
            else:
                used_index.add(index/step)
                index/=step
    if len(approx)!=4:
        return []
    return approx

def _bound(value, upper, lower):
    return min(max(value, lower), upper)

def rearange_vertices(vertices, img_shape):
    sorted_vertices_y = sorted(vertices, key=lambda x: x[0,1])
    sorted_vertices_x = sorted(vertices, key=lambda x: x[0,0])
    mid_y = (sorted_vertices_y[2][0][1]-sorted_vertices_y[1][0][1])
    max_y = (sorted_vertices_y[3][0][1]-sorted_vertices_y[0][0][1])
    mid_x = (sorted_vertices_x[2][0][0]-sorted_vertices_x[1][0][0])
    max_x = (sorted_vertices_x[3][0][0]-sorted_vertices_x[0][0][0])
    
    if (mid_y/max_y) > (mid_x/max_x):
        sorted_vertices = sorted_vertices_y
        if sorted_vertices[0][0][0]<sorted_vertices[1][0][0]:
            nw = sorted_vertices[0][0]
            ne = sorted_vertices[1][0]
        else:
            nw = sorted_vertices[1][0]
            ne = sorted_vertices[0][0]
        if sorted_vertices[2][0][0]<sorted_vertices[3][0][0]:
            sw = sorted_vertices[2][0]
            se = sorted_vertices[3][0]
        else:
            sw = sorted_vertices[3][0]
            se = sorted_vertices[2][0]
    else:
        sorted_vertices = sorted_vertices_x
        if sorted_vertices[0][0][1]<sorted_vertices[1][0][1]:
            nw = sorted_vertices[0][0]
            sw = sorted_vertices[1][0]
        else:
            nw = sorted_vertices[1][0]
            sw = sorted_vertices[0][0]
        if sorted_vertices[2][0][1]<sorted_vertices[3][0][1]:
            ne = sorted_vertices[2][0]
            se = sorted_vertices[3][0]
        else:
            ne = sorted_vertices[3][0]
            se = sorted_vertices[2][0]
    diagonal_length = ((se[0]-nw[0])**2+(se[1]-nw[1])**2)**0.5
    diagonal_length+= ((ne[0]-sw[0])**2+(ne[1]-sw[1])**2)**0.5
    diagonal_length/= 2
    extension = diagonal_length*0.05
    
    ####################################
    # 真正输出的时候不要加这个 extension
    # 让 recognition 算法自己来添加该部分
    ####################################
    
    return [(_bound(se[0]+extension, img_shape[0], 0), _bound(se[1]+extension, img_shape[1], 0)),
            (_bound(sw[0]-extension, img_shape[0], 0), _bound(sw[1]+extension, img_shape[1], 0)),
            (_bound(nw[0]-extension, img_shape[0], 0), _bound(nw[1]-extension, img_shape[1], 0)),
            (_bound(ne[0]+extension, img_shape[0], 0), _bound(ne[1]-extension, img_shape[1], 0)),]

def crop_out_plate(img, vertices):
    pts1 = np.float32(vertices)
    pts2 = np.float32([[300,150],[0,150],[0,0],[300,0]])
    M=cv2.getPerspectiveTransform(pts1,pts2)
    return cv2.warpPerspective(np.uint8(img),M,(300,150))

class detecter:
    
    def __init__(self, model_path):
        tf.reset_default_graph()
        self.model = seg_model()
        self.sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(self.sess, model_path)
    
    def predict(self, imgs):
        """
        img channels should RGB
        """
        x_shape = (512,512)
        xs = []
        for img in imgs:
            if np.max(img)>1:
                x = cv2.resize(img/255.1, x_shape)
            else:
                x = cv2.resize(x.astype(float), x_shape)
            xs.append(x)
        prediction = self.model.predict(self.sess, xs)
        result = []
        for i,p in enumerate(prediction):
            current_p = cv2.resize(p, (img.shape[1], img.shape[0]))
            vertices_p = seg_to_vertices((current_p>0.5).astype(float))
            if len(vertices_p)>0:
                vertices_p = rearange_vertices(vertices_p, (img.shape[1], img.shape[0]))
                plate_p = crop_out_plate(imgs[i], vertices_p)
            else:
                plate_p = np.zeros([150,300])
            result.append(plate_p)
        return result