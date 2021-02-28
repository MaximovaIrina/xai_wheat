import os
import json
import shutil
import numpy as np
import cv2 as cv


def createStruct(src_root, dst_root, prop):
    container, spectrum, angle = prop.split('_')
    new_folder = os.path.join(dst_root, prop)
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)
        days = os.listdir(src_root)
        days_paths = [os.path.join(src_root, day, container, spectrum, angle) for day in days]
        for day in days_paths:
            old_paths = [os.path.join(day, file) for file in os.listdir(day)]
            new_paths = [os.path.join(new_folder, file) for file in os.listdir(day)]
            [shutil.copyfile(src, dst) for src, dst in zip(old_paths, new_paths)]
        print(f'Create {new_folder} with {container}_{spectrum}_{angle} images')
    else:
        print(f'Folder {new_folder} already exists')
    return


def renameRGB(root):
    imgs = os.listdir(root)
    for img in imgs:
        src = os.path.join(root, img)
        img = img.replace('_Реальное изображение', '')
        dst = os.path.join(root, img)
        os.rename(src, dst)
    return


def createDatasetJson(root):
    ir_png_root = os.path.join(root, prop, 'IR')
    ir_paths = [os.path.join(ir_png_root, img) for img in os.listdir(ir_png_root)]

    rgb_png_root = os.path.join(root, prop, 'RGB')
    renameRGB(rgb_png_root)
    rgb_paths = [os.path.join(rgb_png_root, img) for img in os.listdir(rgb_png_root)]

    data = {'IR': ir_paths, 'RGB': rgb_paths}
    return data


def normalization(ir, rgb, coef, b_h, b_w):
    h, w, _ = rgb.shape
    if h == 300 and w == 400:
        return ir, rgb

    h_, w_ = int(h * coef), int(w * coef)
    ir = ir[:, :1300, ]
    ir = cv.resize(ir, (w_, h_))

    h_, w_, _ = ir.shape
    rgb = rgb[(h - h_) // 2 + b_h: (h + h_) // 2 + b_h, (w - w_) // 2 + b_w: (w + w_) // 2 + b_w, ]


    h, w, _ = ir.shape
    ir = ir[100:h - 100, 100:w - 150, :]
    rgb = rgb[100:h - 100, 100:w - 150, :]

    ir = cv.resize(ir, (400, 300))
    rgb = cv.resize(rgb, (400, 300))
    return ir, rgb


def norm_images(data):
    for tir_path, rgb_path in zip(data['IR'], data['RGB']):
        tir = cv.imread(tir_path)
        rgb = cv.imread(rgb_path)
        tir, rgb = normalization(tir, rgb, coef=0.56, b_h=50, b_w=10)
        cv.imwrite(tir_path, tir)
        cv.imwrite(rgb_path, rgb)
    return


def createDatasetDiv2(data, dst_root, prop):
    tir_div_2_path = os.path.join(dst_root, prop, 'IR')
    if not os.path.exists(tir_div_2_path):
        os.makedirs(tir_div_2_path)

    rgb_div_2_path = os.path.join(dst_root, prop, 'RGB')
    if not os.path.exists(rgb_div_2_path):
        os.makedirs(rgb_div_2_path)

    for tir_path, rgb_path in zip(data['IR'], data['RGB']):
        tir = cv.imread(tir_path)
        rgb = cv.imread(rgb_path)
        name = tir_path.split('\\')[-1:][0].split('.')[0]

        h, w, _ = np.shape(tir)
        tir_l, tir_r = tir[:, :w // 2 - 10, :], tir[:, w // 2 + 10:, :]
        rgb_l, rgb_r = rgb[:, :w // 2 - 10, :], rgb[:, w // 2 + 10:, :]

        tir_l_name = os.path.join(tir_div_2_path, 'l_' + name + '.png')
        tir_r_name = os.path.join(tir_div_2_path, 'r_' + name + '.png')
        rgb_l_name = os.path.join(rgb_div_2_path, 'l_' + name + '.png')
        rgb_r_name = os.path.join(rgb_div_2_path, 'r_' + name + '.png')

        cv.imwrite(tir_l_name, tir_l)
        cv.imwrite(tir_r_name, tir_r)
        cv.imwrite(rgb_l_name, rgb_l)
        cv.imwrite(rgb_r_name, rgb_r)
    return


if __name__ == '__main__':
    root = os.path.join(os.getcwd(), '../ds/src')
    new_struct = os.path.join(os.getcwd(), '../ds/struct')
    if not os.path.exists(new_struct):
        os.makedirs(new_struct)
    prop = 'box_IR_90'
    createStruct(root, new_struct, prop)

    ''' EXTRACT TIR & RGB FROM IRSOFT TO STRUCT_PNG FOLDER '''

    try:
        struct_png = os.path.join(os.getcwd(), '..', 'ds', 'struct_png')
    except FileNotFoundError:
        print(f'Folder /ds/struct_png doesn''t exists. You must GET IR IMAGE AND REAL IMAGE AS PNG ')
    data = createDatasetJson(struct_png)
    with open('dataset.json', 'w') as json_file:
        json.dump(data, json_file)

    norm_images(data)

    div2_path = os.path.join(os.getcwd(), '../ds/dev2')
    if not os.path.exists(new_struct):
        os.makedirs(div2_path)
    createDatasetDiv2(data, div2_path, prop)
    data_div2 = createDatasetJson(div2_path)
    with open('dataset_dev2.json', 'w') as json_file:
        json.dump(data_div2, json_file)








