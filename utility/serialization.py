
from PIL import Image
import numpy as np

import json
import os
import pickle
from logger import Logger

import cv2

log = Logger(__name__)
log.logger.propagate = False


class NumpyArrayEncoder(json.JSONEncoder):
    """Assistant class for serialize the numpy to json.
    
    Convert numpy to string list.
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyArrayEncoder, self).default(obj)


def cam_param_dict2json(camera_param_data, json_file_path):
    """ Save the camera parameters to json file.

    :param camera_param_data: camera parameters.
    :type camera_param_data: dict
    :param json_file_path: output json file's path.
    :type json_file_path: str
    """
    with open(json_file_path, 'w') as fp:
        json.dump(camera_param_data, fp,  cls=NumpyArrayEncoder, indent=4)


def cam_param_json2dict(json_file_path):
    """Load the camera parameters form json file.

    Convert all parameters to numpy array.

    :param json_file_path: the json file path.
    :type json_file_path: str
    :return: camera parameter
    :rtype: dict
    """
    dict_data = None
    with open(json_file_path) as json_file:
        dict_data = json.load(json_file)

    def _cam_param_json2dict(dict_data):
        for key, value in dict_data.items():
            if isinstance(value, list):
                dict_data[key] = np.asarray(value)
            elif isinstance(value, dict):
                dict_data[key] = _cam_param_json2dict(value)
        return dict_data

    # parer dict and translate the list to numpy array
    _cam_param_json2dict(dict_data)

    return dict_data


def save_cam_params(json_file_path, face_index_list, cam_params_list):
    """Save sub-images' camera parameters

    :param face_index_list: The available faces index list.
    :type face_index_list: list
    :param cam_params_list: The 20 faces' camera parameters.
    :type cam_params_list: list
    """
    camera_param_data = {}
    # camera parameter list to dict
    for face_index in face_index_list:
        camera_param_data[face_index] = cam_params_list[face_index]

    # dict to json file
    with open(json_file_path, 'w') as fp:
        json.dump(camera_param_data, fp,  cls=NumpyArrayEncoder, indent=4)


def load_cam_params(json_file_path):
    """Load sub-images; camera parameters form file.
    """
    # load json to dict
    with open(json_file_path) as json_file:
        dict_data = json.load(json_file)

    # dict to list
    cam_params_list = []
    for index in dict_data.keys():
        cam_params_list.append(dict_data[index])

    return cam_params_list


def get_sha256(data):
    """Return a SHA-256 hash of the given data array.

    :param data: the binary data array
    :type data: numpy or str
    """
    if data is None:
        log.warn("get_sha256 input data is None!")
        return None
        
    import hashlib
    if isinstance(data, str):
        return hashlib.sha256(data.encode('utf-8')).hexdigest()
    elif isinstance(data, np.ndarray):
        return hashlib.sha256(data.data).hexdigest()
    else:
        log.error("current do not support hash {}".format(type(data)))


def pixel_corresponding_save(json_file_path,
                             src_image_filename, src_image_data,
                             tar_image_filename, tar_image_data, pixel_corresponding):
    """ The relationship of pixel corresponding.
    The origin point on the top-left of image.

    ```
    {
        "src_image": "001.jpg",
        "src_image_sha256": image_numpy_data_sha256,
        "tar_image": "erp.jpg",
        "tar_image_sha256": image_numpy_data_sha256,
        "pixel_corresponding": [
            [src_row_number_0, src_column_number_0, tar_row_number_0, tar_column_number_0],
            [src_row_number_1, src_column_number_1, tar_row_number_1, tar_column_number_1],
        ]
    }
    ```

    :param json_file_path: output json file's path.
    :type json_file_path: str
    :param src_image_filename: source image filename
    :type src_image_filename: str
    :param tar_image_filename: target image filename
    :type tar_image_filename: str
    :param pixel_corresponding: the pixels corresponding relationship, shape is [corresponding_number, 4]
    :type pixel_corresponding: numpy
    """
    json_data = {}

    json_data["src_image_filename"] = os.path.basename(src_image_filename)
    json_data["src_image_sha256"] = get_sha256(src_image_data)
    json_data["tar_image_filename"] = os.path.basename(tar_image_filename)
    json_data["tar_image_sha256"] = get_sha256(tar_image_data)
    json_data["pixel_corresponding_number"] = pixel_corresponding.shape[0]
    json_data["pixel_corresponding"] = pixel_corresponding

    with open(json_file_path, 'w') as fp:
        json.dump(json_data, fp,  cls=NumpyArrayEncoder, indent=4)


def pixel_corresponding_load(json_file_path):
    """
    Load the pixels corresponding relationship from JSON file.
    """
    dict_data = {}
    with open(json_file_path) as json_file:
        dict_data = json.load(json_file)

    def _cam_param_json2dict(dict_data):
        for key, value in dict_data.items():
            if isinstance(value, list):
                dict_data[key] = np.asarray(value)
            elif isinstance(value, dict):
                dict_data[key] = _cam_param_json2dict(value)
        return dict_data

    # parer dict and translate the list to numpy array
    _cam_param_json2dict(dict_data)

    return dict_data


def save_subimages_data(data_dir, filename_prefix,  subimage_list, cam_param_list, pixels_corr_dict,
                        output_corr2file=True):
    """
    Save all subimages data to file, including image, camera parameters and pixels corresponding.

    :param data_dir: the root directory of output file.
    :type data_dir: str
    :param data_dir: the filename's prefix
    :type data_dir: str
    :param subimage_list: [description]
    :type subimage_list: [type]
    :param cam_param_list: [description]
    :type cam_param_list: [type]
    :param pixels_corr_dict: its structure is {1:{2:np.array, 3:np.array, ....}, 2:{1:array, 3:array, ....}....}
    :type pixels_corr_dict: [type]
    """
    subimage_disp_filepath_expression = filename_prefix + "_disp_{:03d}.pfm"
    subimage_filepath_expression = filename_prefix + "_rgb_{:03d}.jpg"
    subimage_param_filepath_expression = filename_prefix + "_cam_{:03d}.json"
    pixels_corresponding_json_filepath_expression = filename_prefix + "_corr_{:03d}_{:03d}.json"

    subimage_number = len(subimage_list)
    if cam_param_list is None:
        log.warn("Camera parameters is empty!")
    elif len(cam_param_list) != subimage_number:
        log.error("The subimage information is not completetd!")

    for src_image_index in range(0, subimage_number):
        # output subimage
        subimage_filepath = data_dir + subimage_filepath_expression.format(src_image_index)
        Image.fromarray(subimage_list[src_image_index].astype(np.uint8)).save(subimage_filepath)

        log.debug("Output image {} pixel corresponding relationship.".format(src_image_index))

        # output camera parameters
        if cam_param_list is not None:
            camparam_filepath = data_dir + subimage_param_filepath_expression.format(src_image_index)
            cam_param_dict2json(cam_param_list[src_image_index], camparam_filepath)

        # output pixel corresponding
        pixels_corr_list = pixels_corr_dict[src_image_index]
        for ref_image_index in pixels_corr_list.keys():
            pixel_corr_filepath = data_dir + pixels_corresponding_json_filepath_expression.format(src_image_index, ref_image_index)

            subimage_src_filepath = subimage_disp_filepath_expression.format(src_image_index)
            subimage_tar_filepath = subimage_disp_filepath_expression.format(ref_image_index)

            if output_corr2file:
                pixel_corresponding_save(pixel_corr_filepath,
                                         subimage_src_filepath, subimage_list[src_image_index],
                                         subimage_tar_filepath, subimage_list[ref_image_index], pixels_corr_list[ref_image_index])


def load_subimages_data():
    """
    Load all subimage data from file, including image, camera parameters and pixels corresponding.
    """
    pass


def subimage_alignment_params(json_file_path, coeffs_scale, coeffs_offset, submap_index_list):
    """ Save disparity maps alignment coefficients.

    :param json_file_path: Coefficients output json file's path.
    :type json_file_path: str
    :param coeffs_scale: the 20 subimage's scale coefficients list.
    :type coeffs_scale: list
    :param coeffs_offset: the 20 subimage's offset coefficients list.
    :type coeffs_offset: list
    :param submap_index_list: the available subimage's index list.
    :type submap_index_list: list
    """
    if len(coeffs_offset) != len(submap_index_list) or len(coeffs_scale) != len(submap_index_list):
        raise RuntimeError("The alignment coefficient is not ")

    # create coefficients dict
    coeffs_dict = {}
    coeffs_dict["storage_order"] = "row_major"
    for index in range(0, len(submap_index_list)):
        data_term_scale = {}
        data_term_scale["coeff_type"] = "scale"
        data_term_scale["filename"] = "face {} alignment scale matrix".format(submap_index_list[index])
        data_term_scale["mat_width"] = coeffs_scale[index].shape[0]
        data_term_scale["mat_hight"] = coeffs_scale[index].shape[1]
        data_term_scale["mat_data"] = coeffs_scale[index]
        subimage_coeff_mat_name = "coeff_mat_" + str(index * 2)
        coeffs_dict[subimage_coeff_mat_name] = data_term_scale

        data_term_offset = {}
        data_term_offset["coeff_type"] = "offset"
        data_term_offset["filename"] = "face {} alignment offset matrix".format(submap_index_list[index])
        data_term_offset["mat_width"] = coeffs_offset[index].shape[0]
        data_term_offset["mat_hight"] = coeffs_offset[index].shape[1]
        data_term_offset["mat_data"] = coeffs_offset[index]
        subimage_coeff_mat_name = "coeff_mat_" + str(index * 2 + 1)
        coeffs_dict[subimage_coeff_mat_name] = data_term_offset

    # output to json
    with open(json_file_path, 'w') as fp:
        json.dump(coeffs_dict, fp,  cls=NumpyArrayEncoder, indent=4)


def save_dispmapalign_intermediate_data(filepath, file_format, **data):
    """
    Save the data used to align disparity maps to file.

    # TODO support "msgpack" format, which is more safe and secure.
    
    :param filepath: the output file's path.
    :type filepath: str
    :param file_format: the output file format, "pickle", "msg"
    :type file_format: str
    :param data: the data to be serialized.
    :type data: dict
    """
    if file_format == "pickle":
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    else:
        raise RuntimeError(f"File format '{file_format}' is not supported")


def load_dispmapalign_intermediate_data(filepath, file_format):
    """
    Load the from disk to align disparity maps to file.

    :param filepath: the output file's path.
    :type filepath: str
    :param file_format: the output file format, "pickle", "msg".
    :type file_format: str
    """
    if file_format == "pickle":
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    else:
        raise RuntimeError(f"File format '{file_format}' is not supported")


def hue_to_rgb(hue):
    if isinstance(hue, np.ndarray):
        hue = 1.0 - hue
        rgb = np.zeros((hue.shape[0], hue.shape[1], 3))
        rgb[..., 0] = hue * 6.0
        rgb[..., 1] = hue * 6.0 + 4.0
        rgb[..., 2] = hue * 6.0 + 2.0
 
    else:
        hue = 1.0 - hue
        rgb = np.zeros(3)
        rgb[0] = hue * 6.0
        rgb[1] = hue * 6.0 + 4.0
        rgb[2] = hue * 6.0 + 2.0

    rgb = np.abs(np.mod(rgb, 6.0) - 3.0) - 1.0
    rgb = np.clip(rgb, 0.0, 1.0)
    return rgb


def heat_to_rgb(heat):
    return hue_to_rgb( 1.0 - heat * 0.65 )


def depth_to_rgb(depth):
    rgb = np.zeros((depth.shape[0], depth.shape[1], 3))
    rgb[..., 0] = depth
    rgb[..., 1] = depth
    rgb[..., 2] = depth
    return rgb


def save_predictions(output_file, erp_rgb_image_data, estimated_depthmap, persp_monodepth, save_npy=False):
    for key in estimated_depthmap.keys():
        path = "{}_{}_{}.png".format(output_file, persp_monodepth, key)
        depth = estimated_depthmap[key]
        depth = depth.astype(np.float64)
        depth_max = depth.max()
        depth_min = depth.min()
        print("depth min: ", depth_min)
        print("depth max: ", depth_max)
        depth = (depth - depth_min) / (depth_max - depth_min)

        depth = 1.0-depth

        cv2.imwrite(path, (heat_to_rgb(depth) * 255 ).astype(np.uint8))

        if save_npy:
            depth *= (depth_max - depth_min)
            np.save(path.replace(".png", ".npy"), depth)
        
    # plt.imsave("{}_rgb.png".format(output_file), erp_rgb_image_data)


