import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import sys
from PIL import Image


def read_imgs_path(imgs_path_para):
    """
    
    :param imgs_path_para: 
    :return: 
    """

    imgs_list_para = [imgs_path_para+s for s in os.listdir(imgs_path_para)]

    return imgs_list_para


def resize_image(imgs_list_para, resize_path_para):
    """
    resize images (whose path is stored in imgs_path_para), and store resized iamges to file(resize_path)
    :param imgs_list_para: 
    :param resize_path_para: 
    :return: 
    """

    i = 0
    for img_path in imgs_list_para:
        # print(img_path)
        outfile = resize_path_para + os.path.splitext(os.path.basename(img_path))[0] + '.jpg'
        try:
            img_origin = Image.open(img_path)
            img_resized = img_origin.resize((128, 128), Image.ANTIALIAS)
            img_resized.save(outfile)
        except IOError:
            print "cannot create thumbnail for '%s'" % img_path

        sys.stdout.write('\r>> Resizing image %d/%d' % (
            i + 1, len(imgs_list_para)))
        sys.stdout.flush()
        print('')
        i = i + 1


def read_image(img_path_para):
    """
    read image
    :param img_path_para: 
    :return:  img in grayscale
    """

    img_para = cv2.imread(img_path_para, cv2.IMREAD_COLOR)
    # cv2.imshow('origin', img)
    img_para = cv2.cvtColor(img_para, cv2.COLOR_BGR2GRAY)
    #  cv2.imshow('gray', gray)

    # plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
    # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    # # plt.plot([200,300,400],[100,200,300],'c', linewidth=5)
    # plt.show()

    return img_para


def sift_features(img_para):
    """
    
    :param img_para: 
    :return: 
    """

    # detector = cv2.xfeatures2d.SIFT_create()
    detector = cv2.xfeatures2d.SURF_create()
    # keypoints = detector.detect(img, None)
    kp_para, des_para = detector.detectAndCompute(img_para, None)

    return kp_para, des_para


def features_to_txt(path_to_write, features_para):
    # with open(path_to_write, 'a') as f:
    #     f.write(features_para)
    np.savetxt(path_to_write, features_para)


def draw_img(img_para, kp_para):
    """

    :param img_para: 
    :param kp_para: 
    :return: 
    """

    # img = cv2.drawKeypoints(img, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img_para = cv2.drawKeypoints(img_para, kp_para, img_para)
    # cv2.imwrite('sift_keypoints.jpg', img)

    cv2.imshow('SURF_features', img_para)
    cv2.waitKey()
    cv2.destroyAllWindows()


def img_to_features(imgs_list_para, output_path_para):
    """
    
    :param imgs_list_para: 
    :param output_path_para: 
    :return: 
    """

    min_features_num = 10000
    i = 0

    for img_path in imgs_list_para:
        img = read_image(img_path)
        kp, des = sift_features(img)

        min_features_num = np.minimum(min_features_num, des.shape[0])
        # print(des.shape)
        features_to_txt(output_path_para + os.path.splitext(os.path.basename(img_path))[0] + '.txt', des)
        # draw_img(img, kp)

        sys.stdout.write('\r>> Converting image %d/%d' % (
            i + 1, len(imgs_list_para)))
        sys.stdout.flush()
        print('')
        i = i + 1

    return min_features_num


if __name__ == '__main__':
    img_list = ['/home/lenovo/Downloads/ml/ic-data/train_resize/23.jpg']

    # # read images path
    # imgs_path = '/home/lenovo/Downloads/ml/ic-data/train/'
    # imgs_list = read_imgs_path(imgs_path)
    # print(len(imgs_list))
    #
    # # resize images
    # imgs_resized_path = '/home/lenovo/Downloads/ml/ic-data/train_resize/'
    # resize_image(imgs_list, imgs_resized_path)

    ########################## first run code above ##############################

    imgs_resized_path = '/home/lenovo/Downloads/ml/ic-data/train_resize/'
    imgs_resized_list = read_imgs_path(imgs_resized_path)
    print(len(imgs_resized_list))

    features_output_path = '/home/lenovo/Downloads/ml/ic-data/train_features/'
    min_fn = img_to_features(imgs_resized_list, features_output_path)
    print(min_fn)
