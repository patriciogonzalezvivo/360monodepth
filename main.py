from utility import fs_utility
from utility.fs_utility import FileNameConvention

import os
import re
from pathlib import Path
import numpy as np
from PIL import Image
import shutil
import argparse

from utility import depthmap_align, image_io
from utility import depthmap_utils
from utility import blending
from utility import serialization
from utility import projection_icosahedron as proj_ico

import warnings
warnings.filterwarnings("ignore")

from utility.logger import Logger

log = Logger(__name__)
log.logger.propagate = False


def grid_size_type(arg_value, pat=re.compile(r"^[2-9]+[xX][2-9]+$")):
    if not pat.match(arg_value):
        raise argparse.ArgumentTypeError
    return arg_value


class Options():
    def __init__(self) -> None:
        self.parser = None
        self.input_image_path = None
        self.output_image_path = None
        self.save_npy = False

        # 0) input data option
        self.available_steps = [3, 4, 5]

        # 1) subimage generation option
        self.subimage_available_list = list(range(0, 20))
        self.subimage_padding_size = 0.3
        self.subimage_tangent_image_width = 400
        self.persp_monodepth = "midas2"

        # 3) subimage depthmap alignment parameters
        self.dispalign_corr_thread_number = 10
        self.dispalign_pyramid_layer_number = 1
        self.multi_res_grid = False
        self.dispalign_pixelcorr_downsample_ratio = 0.001
        self.dispalign_iter_num = 100
        self.dispalign_ceres_max_linear_solver_iterations = 10
        self.dispalign_method = "group"
        self.dispalign_weight_project = 1.0
        self.dispalign_weight_smooth = 40.
        self.dispalign_weight_scale = 0.007
        self.coeff_fixed_face_index = -1
        self.dispalign_align_coeff_grid_width = 8
        self.dispalign_align_coeff_grid_height = 7
        self.dispalign_output_dir = None
        self.dispalign_debug_enable = False

        # 2) subimage blending option
        self.blending_method = None

        # 3) debug option
        self.debug_enable = False
        self.debug_output_dir = None

        # 4) post-process
        self.rm_debug_folder = False

        # 5) dataset configuration
        self.dataset_matterport_hexagon_mask_enable = False
        # the circumradius in gnomonic coordinate system, 
        self.dataset_matterport_blur_area_height = 0
        self.dataset_matterport_blurarea_shape = "circle"  # "hexagon",  "circle"

    def parser_arguments(self, parser):
        self.parser = parser

        # 1) parser CLI arguments
        parser.add_argument("--input", "-i", type=str, help="experiment name", required=True)
        parser.add_argument("--output", "-o", type=str, default="None", help="experiment name")

        parser.add_argument("--blending_method", type=str, default="frustum", choices=['poisson', 'frustum', 'radial', 'nn', 'mean', 'all'])
        
        parser.add_argument("--grid_size", type=grid_size_type, default="8x7", help="width x height")
        parser.add_argument("--padding", type=float, default="0.3")
        parser.add_argument("--multires_levels", type=int, default=1, help="Levels of multi-resolution pyramid. If > 1 then --grid_size is the lowest resolution")
        parser.add_argument("--persp_monodepth", type=str, default="midas2", choices=["midas2", "midas3", "zoedepth"])
        parser.add_argument('--depthalignstep', type=int, nargs='+', default=[1, 2, 3, 4])
        parser.add_argument("--rm_debug_folder", default=True, action='store_false')
        parser.add_argument("--intermediate_data", default=False, action='store_true', help="save intermediate data generated during the pipeline")
        parser.add_argument("--npy", default=False, action='store_true', help="save npy file")
        opt_arguments = parser.parse_args()

        # 2) update options
        self.blending_method = opt_arguments.blending_method
        self.input_image_path = opt_arguments.input
        self.save_npy = opt_arguments.npy

        if opt_arguments.output == "None":
            folder = os.path.dirname(opt_arguments.input)
            filename_wout_ext = os.path.splitext(os.path.basename(opt_arguments.input))[0]
            self.output_image_path = os.path.join(folder, filename_wout_ext)
        else:
            self.output_image_path = opt_arguments.output

        print("Output file: {}".format(self.output_image_path))
              
        self.rm_debug_folder = opt_arguments.rm_debug_folder
        self.debug_enable = opt_arguments.intermediate_data
        self.dispalign_debug_enable = opt_arguments.intermediate_data
        self.subimage_padding_size = opt_arguments.padding
        self.dispalign_align_coeff_grid_width = int(opt_arguments.grid_size.lower().split("x")[0])
        self.dispalign_align_coeff_grid_height = int(opt_arguments.grid_size.lower().split("x")[1])
        self.dispalign_pyramid_layer_number = opt_arguments.multires_levels
        if self.dispalign_pyramid_layer_number > 1:
            self.dispalign_iter_num = 50
            self.multi_res_grid = True
        self.persp_monodepth = opt_arguments.persp_monodepth
        self.available_steps = opt_arguments.depthalignstep

        self.print()

    def print(self):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- Options End -------------------'
        print(message)


def depthmap_estimation(erp_rgb_image_data, fnc, opt, blendIt):
    """ Estimate the ERP image depth map from ERP rgb image.

    :param erp_rgb_image_data: RGB image data, [height, width, 3]
    :type erp_rgb_image_data: numpy
    :param filenameconv: the file name convention object.
    :type filenameconv: dict
    :return: the estimated depth map, [height, width]
    :rtype: numpy
    """

    erp_image_height = erp_rgb_image_data.shape[0]
    subimage_dispmap_erp_list = []
    subimage_depthmap_erp_list = []  # the depth map in ERP image space
    subimage_rgb_list = []
    dispmap_aligned_list = []
    subimage_cam_param_list = []
    tangent_image_gnomo_xy = []  # to convert the perspective image to ERP image

    # 1) load ERP image & project to 20 face images
    if 1 in opt.available_steps:
        log.info("1) load ERP image & project to 20 face images")
        # project to 20 images
        subimage_rgb_list, _, points_gnomocoord = proj_ico.erp2ico_image(erp_rgb_image_data,
                                                                         opt.subimage_tangent_image_width,
                                                                         opt.subimage_padding_size,
                                                                         full_face_image=True)
        tangent_image_gnomo_xy = points_gnomocoord[1]

        if opt.debug_enable:
            log.debug("Debug enable, Step 1 output subimage rgb to {}".format(fnc.root_dir))
            for index in range(0, len(subimage_rgb_list)):
                src_image_output_path = fnc.subimage_rgb_filename_expression.format(index)
                Image.fromarray(subimage_rgb_list[index].astype(np.uint8)).save(src_image_output_path)
                if index % 4 == 0:
                    print("Output subimage to {}.".format(src_image_output_path))

            # output subimage's rgb to image array
            subimage_rgb_array_filepath = fnc.subimage_rgb_filename_expression.format(999)
            image_io.subimage_save_ico(subimage_rgb_list, subimage_rgb_array_filepath)

    # 2) MiDaS estimate disparity maps
    if 2 in opt.available_steps:
        log.info("2) MiDaS estimate disparity maps")
        # load subimage rgb data from disk
        if not subimage_rgb_list or not tangent_image_gnomo_xy:
            log.info("load subimage's rgb data from disk.")
            for index in list(range(0, 20)):
                src_image_output_path = fnc.subimage_rgb_filename_expression.format(index)
                subimage_rgb_list.append(np.asarray(Image.open(src_image_output_path)))
            log.info("generate face gnomonic coordinate")
            _, _, points_gnomocoord = proj_ico.erp2ico_image(erp_rgb_image_data, opt.subimage_tangent_image_width,
                                                             opt.subimage_padding_size, full_face_image=True)
            tangent_image_gnomo_xy = points_gnomocoord[1]

        subimage_depthmap_persp_list = []
        # estimate disparity map
        subimage_dispmap_persp_list = depthmap_utils.run_persp_monodepth(subimage_rgb_list, opt.persp_monodepth)
        # convert disparity map to depth map
        for dispmap_persp in subimage_dispmap_persp_list:
            subimage_depthmap_persp_list.append(depthmap_utils.disparity2depth(dispmap_persp))
        # convert each subimage's perspective depth map to ERP depth map.
        for depthmap_persp in subimage_depthmap_persp_list:
            subimage_depthmap_erp_list.append(
                depthmap_utils.subdepthmap_tang2erp(depthmap_persp, tangent_image_gnomo_xy))
        # convert each subimage's from ERP depth map to perspective map.
        for depthmap_erp in subimage_depthmap_erp_list:
            subimage_dispmap_erp_list.append(depthmap_utils.depth2disparity(depthmap_erp).astype(np.float32))

        # output disparity map and visualized result.
        if opt.debug_enable:
            # output disparity map array
            dispmap_array_filename = fnc.subimage_dispmap_persp_filename_expression.format(999)
            depthmap_utils.depth_ico_visual_save(subimage_dispmap_persp_list, dispmap_array_filename + ".jpg")
            depthmap_persp_array_filename = fnc.subimage_depthmap_persp_filename_expression.format(999)
            depthmap_utils.depth_ico_visual_save(subimage_depthmap_persp_list, depthmap_persp_array_filename + ".jpg")
            depthmap_erp_array_filename = fnc.subimage_depthmap_erp_filename_expression.format(999)
            depthmap_utils.depth_ico_visual_save(subimage_depthmap_erp_list, depthmap_erp_array_filename + ".jpg")
            subimage_dispmap_erp_filepath = fnc.subimage_dispmap_erp_filename_expression.format(999)
            depthmap_utils.depth_ico_visual_save(subimage_dispmap_erp_list, subimage_dispmap_erp_filepath + ".jpg")

            # output disparity map
            for index in range(0, len(subimage_rgb_list)):
                depth_filename = fnc.subimage_dispmap_erp_filename_expression.format(index)
                depthmap_utils.write_pfm(depth_filename, subimage_dispmap_erp_list[index], scale=1)
                depthmap_utils.depth_visual_save(subimage_dispmap_erp_list[index], depth_filename + ".jpg")
                if index % 4 == 0:
                    print("Output subimages disparity map to {}".format(depth_filename))

    # 3) align disparity maps
    if 3 in opt.available_steps:
        log.info("3) align disparity maps")
        # load subimage disparity map from disk
        if not subimage_dispmap_erp_list or not subimage_rgb_list:
            log.info("load subimage's MiDaS disparity maps from disk.")
            for index in list(range(0, 20)):
                src_image_output_path = fnc.subimage_rgb_filename_expression.format(index)
                subimage_rgb_list.append(np.asarray(Image.open(src_image_output_path)))
                depth_filename = fnc.subimage_dispmap_erp_filename_expression.format(index)
                subimage_dispmap_erp_list.append(depthmap_utils.read_pfm(depth_filename)[0])

        # subimages dispmap align parameters
        # set alignment parameter
        depthmap_aligner = depthmap_align.DepthmapAlign()
        depthmap_aligner.opt = opt  # the global configuration
        depthmap_aligner.pyramid_layer_number = opt.dispalign_pyramid_layer_number
        depthmap_aligner.multi_res_grid = opt.multi_res_grid
        depthmap_aligner.downsample_pixelcorr_ratio = opt.dispalign_pixelcorr_downsample_ratio
        depthmap_aligner.align_method = opt.dispalign_method
        depthmap_aligner.ceres_max_num_iterations = opt.dispalign_iter_num
        depthmap_aligner.weight_project = opt.dispalign_weight_project
        depthmap_aligner.weight_smooth = opt.dispalign_weight_smooth
        depthmap_aligner.weight_scale = opt.dispalign_weight_scale
        depthmap_aligner.coeff_fixed_face_index = opt.coeff_fixed_face_index
        depthmap_aligner.align_coeff_grid_width = opt.dispalign_align_coeff_grid_width
        depthmap_aligner.align_coeff_grid_height = opt.dispalign_align_coeff_grid_height
        depthmap_aligner.ceres_max_linear_solver_iterations = opt.dispalign_ceres_max_linear_solver_iterations
        if opt.dispalign_output_dir is not None:
            depthmap_aligner.output_dir = opt.dispalign_output_dir  # output cpp module alignment coefficient
        else:
            depthmap_aligner.output_dir = fnc.root_dir
        depthmap_aligner.debug = opt.dispalign_debug_enable
        depthmap_aligner.subimages_rgb = subimage_rgb_list  # original 20 subimages

        # set the output file name for debug, if not set do not output debug files.
        # depthmap_aligner.subimage_alignment_intermedia_filepath_expression = fnc.subimage_alignment_intermedia_filename_expression
        depthmap_aligner.subimage_pixelcorr_filepath_expression = fnc.subimage_pixelcorr_filename_expression
        # depthmap_aligner.subimage_warpedimage_filepath_expression = fnc.subimage_warpedimage_filename_expression
        # depthmap_aligner.subimage_warpeddepth_filename_expression = fnc.subimage_warpeddepth_filename_expression
        # depthmap_aligner.subimage_depthmap_aligning_filepath_expression = fnc.subimage_alignment_depthmap_input_filename_expression

        subimage_dispmap_list_sub = [subimage_dispmap_erp_list[i] for i in opt.subimage_available_list]
        dispmap_aligned_list, coeffs_scale, coeffs_offset, subimage_cam_param_list = \
            depthmap_aligner.align_multi_res(erp_rgb_image_data, subimage_dispmap_list_sub, opt.subimage_padding_size,
                                             opt.subimage_available_list)

        if opt.debug_enable:
            # visualized cpp module output aligned subimage's disparity map
            if opt.dispalign_debug_enable:
                for subimage_index in opt.subimage_available_list:
                    subimage_filepath = fnc.subimage_dispmap_cpp_aligned_filename_expression.format(subimage_index)
                    dispmap, _ = depthmap_utils.read_pfm(str(subimage_filepath))
                    dispmap_vis_filepath = subimage_filepath + ".jpg"
                    depthmap_utils.depth_visual_save(dispmap, dispmap_vis_filepath)

            # output the scale ans offset to json
            serialization.subimage_alignment_params(fnc.subimage_dispmap_aligned_coeffs_filename_expression,
                                                    coeffs_scale, coeffs_offset, opt.subimage_available_list)

            # visualize align coefficients to image files
            depthmap_utils.depth_ico_visual_save(coeffs_scale,
                                                 fnc.subimage_dispmap_aligned_coeffs_filename_expression + "_scale.jpg",
                                                 opt.subimage_available_list)
            depthmap_utils.depth_ico_visual_save(coeffs_offset,
                                                 fnc.subimage_dispmap_aligned_coeffs_filename_expression + "_offset.jpg",
                                                 opt.subimage_available_list)

            # output the camera parameters to json files
            serialization.save_cam_params(fnc.subimage_camsparams_list_filename_expression, opt.subimage_available_list,
                                          subimage_cam_param_list)

            # output alignment disparity map array image
            depth_array_filename = fnc.subimage_dispmap_aligned_filename_expression.format(999)
            depthmap_utils.depth_ico_visual_save(dispmap_aligned_list, depth_array_filename + ".jpg",
                                                 opt.subimage_available_list)

            # output visualized alignment disparity map
            for index in range(0, len(dispmap_aligned_list)):
                depth_filename = fnc.subimage_dispmap_aligned_filename_expression.format(
                    opt.subimage_available_list[index])
                depthmap_utils.write_pfm(depth_filename, dispmap_aligned_list[index].astype(np.float32), scale=1)
                depthmap_utils.depth_visual_save(dispmap_aligned_list[index], depth_filename + ".jpg")
                if index % 4 == 0:
                    print("Output aligned subimages disparity map to {}".format(depth_filename))

    # 4) blender to ERP image & output ERP disparity map and visualized image
    if 4 in opt.available_steps:
        if not dispmap_aligned_list or not subimage_cam_param_list:
            log.info("load subimage's aligned disparity maps from disk.")
            for index in opt.subimage_available_list:
                depth_filename = fnc.subimage_dispmap_aligned_filename_expression.format(index)
                if os.path.isfile(depth_filename):
                    dispmap_aligned_list.append(depthmap_utils.read_pfm(depth_filename)[0])
                else:
                    dispmap_aligned_list.append(subimage_dispmap_erp_list[index])
            log.info("load subimage's camera parameters from disk.")

        log.info("4) blender to ERP image")

        erp_dispmap_blend = None
        if len(opt.subimage_available_list) == 20:
            # if idx == 0:
            blendIt.tangent_images_coordinates(erp_image_height, dispmap_aligned_list[0].shape)
            blendIt.erp_blendweights(subimage_cam_param_list, erp_image_height, dispmap_aligned_list[0].shape)
            blendIt.compute_linear_system_matrices(erp_image_height, erp_image_height * 2, blendIt.frustum_blendweights)
            erp_dispmap_blend = blendIt.blend(dispmap_aligned_list, erp_image_height)
        else:
            # available faces number is not 20, output subimages linear blend result
            log.warn("Linear blend {} subimages.".format(len(opt.subimage_available_list)))

            # file with blank depth
            dispmap_aligned_list_filled = depthmap_utils.fill_ico_subimage(dispmap_aligned_list,
                                                                           opt.subimage_available_list)
            erp_dispmap_blend = proj_ico.ico2erp_image(dispmap_aligned_list_filled, erp_image_height, opt.subimage_padding_size, blender_method="mean")

        if opt.debug_enable:
            if opt.blending_method == 'all':
                erp_dispmap_blend_save = erp_dispmap_blend['poisson']
            else:
                erp_dispmap_blend_save = erp_dispmap_blend[opt.blending_method]

            # output blending result disparity map
            erp_aligned_dispmap_filepath = fnc.erp_depthmap_blending_result_filename_expression
            depthmap_utils.write_pfm(erp_aligned_dispmap_filepath, erp_dispmap_blend_save.astype(np.float32), scale=1)
            erp_aligned_dispmap_vis_filepath = fnc.erp_depthmap_vis_blending_result_filename_expression
            depthmap_utils.depth_visual_save(erp_dispmap_blend_save, erp_aligned_dispmap_vis_filepath + ".jpg")

        return erp_dispmap_blend


def monodepth_360(opt):
    """Pipeline."""
    input_image_path = opt.input_image_path
    output_image_path = opt.output_image_path
    if not os.path.exists(input_image_path):
        print("Input file {} does not exist.".format(input_image_path))
        return

    energy_weights = np.array([opt.dispalign_weight_smooth, opt.dispalign_weight_scale])[None, ...]

    # BlendIt object. Equation 7 of the paper
    blend_it = blending.BlendIt(opt.subimage_padding_size, len(opt.subimage_available_list), opt.blending_method)
    blend_it.fidelity_weight = 0.1

    for weights in energy_weights:
        if isinstance(weights, np.ndarray):
            opt.dispalign_weight_smooth = weights[0]
            opt.dispalign_weight_scale = weights[1]
            if opt.dispalign_weight_scale > 0:
                opt.coeff_fixed_face_index = -1
            else:
                opt.coeff_fixed_face_index = 7
        else:
            weights = np.array([weights])
            blend_it.fidelity_weight = weights[0]

        # enable all steps [extraction, midas, alignment, blending]
        times_header = ["t_extraction(s)", "t_midas(s)", "t_alignment(s)", "t_blending(s)"]
        times_header = [times_header[i-1] for i in opt.available_steps]
        times_header.append("t_total(s)")

        if "matterport" in input_image_path:
            opt.dataset_matterport_hexagon_mask_enable = True
            opt.dataset_matterport_blur_area_height = 140  # * 0.75
        else:
            opt.dataset_matterport_hexagon_mask_enable = False


        data_root = os.path.dirname(input_image_path)
        debug_output_dir = os.path.join(data_root, "debug/")
        Path(debug_output_dir).mkdir(parents=True, exist_ok=True)

        erp_image_filepath = input_image_path
        filename_base, _ = os.path.splitext(os.path.basename(input_image_path))

        fnc = FileNameConvention()
        fnc.set_filename_basename(filename_base)
        fnc.set_filepath_folder(debug_output_dir)

        # load ERP rgb image and estimate the ERP depth map
        erp_rgb_image_data = image_io.image_read(erp_image_filepath)
        # Load matrices for blending linear system
        estimated_depthmap = depthmap_estimation(erp_rgb_image_data, fnc, opt, blend_it)

        serialization.save_predictions(output_image_path, erp_rgb_image_data, estimated_depthmap, opt.persp_monodepth, opt.save_npy)

        # Remove temporal storage folder
        if opt.rm_debug_folder and os.path.isdir(debug_output_dir):
            shutil.rmtree(debug_output_dir)


if __name__ == "__main__":
    # parser arguments
    opt = Options()
    parser = argparse.ArgumentParser()
    opt.parser_arguments(parser)

    # estimate depth map from ERP rgb image
    monodepth_360(opt)