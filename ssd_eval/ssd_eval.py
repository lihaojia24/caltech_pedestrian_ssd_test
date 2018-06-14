import os
import sys
import argparse
import numpy as np
from PIL import Image, ImageDraw

caffe_root = '/home/hector/project/proj-ssd/ssd/caffe/'
os.chdir(caffe_root)
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe

class SSDDetection: 
    def __init__(self, gpu_id, model_def, model_weights, image_resize):
        caffe.set_device(gpu_id)
        caffe.set_mode_gpu()

        self.image_resize = image_resize
        self.net = caffe.Net(model_def,
                             model_weights,
                             caffe.TEST)
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2, 0, 1))
        self.transformer.set_mean('data', np.array([104, 117, 123]))
        self.transformer.set_raw_scale('data', 255)
        self.transformer.set_channel_swap('data', (2, 1, 0))
    
    def detect(self, image_file, conf_thresh=0.002, topn=1000):
        self.net.blobs['data'].reshape(1, 3, self.image_resize, self.image_resize)
        image = caffe.io.load_image(image_file)
        image_name = image_file.split('/')[-1].split('.')[0]
        transformed_image = self.transformer.preprocess('data', image)
        self.net.blobs['data'].data[...] = transformed_image
        detections = self.net.forward()['detection_out']
        det_label = detections[0,0,:,1]
        det_conf = detections[0,0,:,2]
        det_xmin = detections[0,0,:,3]
        det_ymin = detections[0,0,:,4]
        det_xmax = detections[0,0,:,5]
        det_ymax = detections[0,0,:,6]
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= conf_thresh]
        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        result = []
        for i in xrange(min(topn, top_conf.shape[0])):
            xmin = top_xmin[i] # xmin = int(round(top_xmin[i] * image.shape[1]))
            ymin = top_ymin[i] # ymin = int(round(top_ymin[i] * image.shape[0]))
            xmax = top_xmax[i] # xmax = int(round(top_xmax[i] * image.shape[1]))
            ymax = top_ymax[i] # ymax = int(round(top_ymax[i] * image.shape[0]))
            score = top_conf[i]
            label = int(top_label_indices[i])
            # result.append([xmin, ymin, xmax, ymax, label, score, label_name])
            result.append([image_name, xmin, ymin, xmax, ymax, score])
        return result

def main(args):
    detection = SSDDetection(args.gpu_id,
                               args.model_def, args.model_weights,
                               args.image_resize)
    
    # print(result)
    file_content = {}
    for file in os.listdir(os.path.join(args.image_dir, 'Images')):
        print(file)
        result = detection.detect(os.path.join(args.image_dir, 'Images', file))
        img = Image.open(os.path.join(args.image_dir, 'Images', file))
        # draw = ImageDraw.Draw(img)
        width, height = img.size
        # img_name = file.split('/')[-1]
        dirs = file.split('_')[:3]
        if not file_content.has_key(dirs[0]):
            file_content[dirs[0]] = {}
        if not file_content[dirs[0]].has_key(dirs[1]):
            file_content[dirs[0]][dirs[1]] = ""
        file_content[dirs[0]][dirs[1]]
        for item in result:
            xmin = int(round(item[1] * width))
            ymin = int(round(item[2] * height))
            xmax = int(round(item[3] * width))
            ymax = int(round(item[4] * height))

            # draw.rectangle([xmin, ymin, xmax, ymax], outline=(255, 0, 0))
            # draw.text([xmin, ymin], item[0] + str(item[-1]), (0, 0, 255))


            file_content[dirs[0]][dirs[1]] += '{},{},{},{},{},{}\n'.format(int(dirs[2][1:])+1, xmin, ymin, xmax-xmin, ymax-ymin, item[-1])
        # img.show('zz')
    for dir1 in file_content:
        if not os.path.exists(os.path.join(args.image_dir,dir1)):
            os.makedirs(os.path.join(args.image_dir,dir1))
        for dir2 in file_content[dir1]:
            f = open(os.path.join(args.image_dir,dir1,dir2)+'.txt', "w")
            f.write(file_content[dir1][dir2])
            f.close()
    # draw = ImageDraw.Draw(img)
    # width, height = img.size
    # file_name = 'result.txt'
    # file_content = ''
    # for item in result:
    #     xmin = int(round(item[1] * width))
    #     ymin = int(round(item[2] * height))
    #     xmax = int(round(item[3] * width))
    #     ymax = int(round(item[4] * height))
    #     file_content += '{},{},{},{},{},{}\n'.format(item[0], xmin, ymin, xmax-xmin, ymax-ymin, item[-1])
        # draw.rectangle([xmin, ymin, xmax, ymax], outline=(255, 0, 0))
        # 
    # f = open(file_name, "w")
    # f.write(file_content)
    # f.close()
    # img.show('zz')

def parse_args():
    '''parse args'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--model_def',
                        default='models/VGGNet/VOC0712/SSD_300x300/deploy.prototxt')
    parser.add_argument('--image_resize', default=300, type=int)
    parser.add_argument('--model_weights',
                        default='models/VGGNet/VOC0712/SSD_300x300/'
                        'VGG_VOC0712_SSD_300x300_iter_120000.caffemodel')
    parser.add_argument('--image_file', default='examples/images/fish-bike.jpg')
    parser.add_argument('--image_dir', default='/home/hector/data/caltech_root/VOC/')
    return parser.parse_args()

if __name__ == '__main__':
    main(parse_args())