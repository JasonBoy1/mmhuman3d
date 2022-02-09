import argparse
import pickle

import mmcv
from mmhuman3d.core.cameras.builder import build_cameras
from mmhuman3d.core.conventions.cameras import convert_world_view
from mmhuman3d.utils.mesh_utils import save_meshes_as_objs
from mmhuman3d.utils.path_utils import prepare_output_path
import numpy as np
import torch
import json

from mmhuman3d.core.conventions.keypoints_mapping import convert_kps
from mmhuman3d.core.visualization import visualize_smpl_pose
from mmhuman3d.models.builder import build_registrant

from mmhuman3d.core.visualization.renderer import render_runner
import os
import glob

osj = os.path.join


def parse_args():
    parser = argparse.ArgumentParser(description='mmhuman3d smplify tool')
    parser.add_argument(
        '--keypoint',
        default=None,
        help=('input file path.'
              'Input shape should be [N, J, D] or [N, M, J, D],'
              ' where N is the sequence length, M is the number of persons,'
              ' J is the number of joints and D is the dimension.'))
    parser.add_argument(
        '--keypoint_src',
        default='coco_wholebody',
        help='the source type of input keypoints')
    parser.add_argument('--config', default='f2a_config.py')
    parser.add_argument('--camera_path', help='smplify config file path')
    parser.add_argument('--image_folder', help='smplify config file path')
    parser.add_argument(
        '--model_path',
        default='/mnt/lustre/share/sugar/SMPLmodels/',
        help='smplify config file path')
    parser.add_argument(
        '--uv_param_path',
        default='/mnt/lustre/share/sugar/smpl_uv.pkl',
        help='smplify config file path')

    parser.add_argument('--num_betas', type=int, default=10)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument(
        '--use_one_betas_per_video',
        default=True,
        type=bool,
        help='use one betas to keep shape consistent through a video')
    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda'],
        default='cuda',
        help='device used for smplify')
    parser.add_argument(
        '--gender',
        choices=['neutral', 'male', 'female'],
        default='neutral',
        help='gender of SMPL model')
    parser.add_argument('--exp_dir', help='tmp dir for writing some results')
    parser.add_argument(
        '--verbose',
        action='store_true',
    )
    parser.add_argument(
        '--visualize',
        type=bool,
        default=True,
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    flow2avatar_config = mmcv.Config.fromfile(args.config)
    assert flow2avatar_config.body_model.type.lower() in ['smpld']

    # set cudnn_benchmark
    if flow2avatar_config.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)
    prepare_output_path(args.exp_dir, path_type='dir', overwrite=True)

    with open(osj(args.exp_dir, 'config.json'), 'w') as f:
        json.dump(dict(flow2avatar_config), f)

    d = dict(
        np.load(
            '/mnt/lustre/share/sunqingping/to_wwj/snapshot_people/preprocessed_data/male-1-casual/male-1-casual.npz'
        ))
    pose = d['pose']
    image_names = os.listdir(
        '/mnt/lustre/share/sunqingping/to_wwj/snapshot_people/preprocessed_data/male-1-casual/images'
    )
    image_names.sort()
    mask_names = [
        im_name.split('.jpg')[0] + '_mask.jpg' for im_name in image_names
    ]
    # image_names = d['imgname']
    image_paths = [
        osj(
            '/mnt/lustre/share/sunqingping/to_wwj/snapshot_people/preprocessed_data/male-1-casual/images',
            im_name) for im_name in image_names
    ]

    mask_paths = [
        osj(
            '/mnt/lustre/share/sunqingping/to_wwj/snapshot_people/preprocessed_data/male-1-casual/masks',
            mask_name) for mask_name in mask_names
    ]
    R = d.get('R', None)
    R = torch.Tensor(R) if R is not None else None
    T = d.get('T', None)
    T = torch.Tensor(T) if T is not None else None
    K = torch.Tensor(d['K'])
    if R is not None and T is not None:
        R, T = convert_world_view(R, T)
    cameras = build_cameras(
        dict(
            type='perspective',
            in_ndc=False,
            convention='opencv',
            K=K,
            R=R,
            T=T,
            resolution=(1080, 1080)))
    cameras = cameras.to(device)

    cameras = cameras.extend(len(image_paths))

    gender = str(d.get('gender', args.gender))

    body_model_type = flow2avatar_config.body_model.type
    body_model_config = dict(
        type=body_model_type,
        gender=gender,
        num_betas=args.num_betas,
        model_path=osj(args.model_path, 'smpl'),
        uv_param_path=args.uv_param_path,
    )

    flow2avatar_config.renderer_uv.update(uv_param_path=args.uv_param_path)
    flow2avatar_config.update(
        device=device,
        verbose=args.verbose,
        experiment_dir=args.exp_dir,
        body_model=body_model_config,
        use_one_betas_per_video=args.use_one_betas_per_video)

    # cameras = None
    # if args.camera_path is not None:
    #     with open(args.camera_path, 'rb') as f:
    #         cameras_config = pickle.load(f)
    #     cameras = build_cameras(cameras_config).to(device)

    if args.keypoint is not None:
        with open(args.keypoint, 'rb') as f:
            keypoints_src = pickle.load(f, encoding='latin1')
            if args.input_type == 'keypoints2d':
                assert keypoints_src.shape[-1] == 2
            elif args.input_type == 'keypoints3d':
                assert keypoints_src.shape[-1] == 3
            else:
                raise KeyError('Only support keypoints2d and keypoints3d')

        keypoints, mask = convert_kps(
            keypoints_src,
            src=args.keypoint_src,
            dst=flow2avatar_config.body_model['keypoint_dst'])
        keypoints_conf = np.repeat(mask[None], keypoints.shape[0], axis=0)

        keypoints = torch.tensor(keypoints, dtype=torch.float32, device=device)
        keypoints_conf = torch.tensor(
            keypoints_conf, dtype=torch.float32, device=device)

        if args.keypoint_type == 'keypoints3d':
            data.update(
                dict(keypoints3d=keypoints, keypoints3d_conf=keypoints_conf))

    flow2avatar = build_registrant(dict(flow2avatar_config))
    # d = np.load(
    #     '/mnt/lustre/wangwenjia/datasets/h36m/S1_Eating_2.60457274/S1_Eating_2.60457274.npz'
    # )

    # d = np.load('/mnt/lustre/wangwenjia/mesh/m2m_smpl.npz')
    # pose_dict = {}
    # for k in d:
    #     pose_dict[k] = torch.Tensor(d[k])[0:1]
    pose_dict = flow2avatar.body_model.tensor2dict(
        torch.Tensor(pose),
        betas=torch.Tensor(d['shape']),
        transl=torch.Tensor(d['transl']))
    init_global_orient = pose_dict['global_orient'].to(
        device)[:len(image_paths)]
    init_transl = pose_dict['transl'].to(
        device)[:len(image_paths)] if pose_dict['transl'] is not None else None
    init_body_pose = pose_dict['body_pose'].to(device)[:len(image_paths)]
    init_betas = pose_dict['betas'].to(
        device)[:len(image_paths)] if pose_dict['betas'] is not None else None

    data = dict(
        image_paths=image_paths,
        silhouette_paths=mask_paths,
        cameras=cameras,
        # return_texture=True,
        # return_mesh=True,
        init_global_orient=init_global_orient,
        init_transl=init_transl,
        init_body_pose=init_body_pose,
        init_betas=init_betas)

    flow2avatar_output = flow2avatar(**data)

    # avatar = flow2avatar_output.pop('meshes')

    pose_dict = flow2avatar.body_model.tensor2dict(
        torch.zeros(1, 72).to(device))
    Tpose_output = flow2avatar.body_model(
        displacement=flow2avatar_output['displacement'],
        texture_image=flow2avatar_output['texture_image'],
        return_mesh=True,
        return_texture=True,
        **pose_dict)

    T_avatar = Tpose_output['meshes']
    save_meshes_as_objs(T_avatar[0], [osj(args.exp_dir, 'T_pose.obj')])
    # get smpl parameters directly from smplify output

    for k, v in flow2avatar_output.items():
        if isinstance(v, torch.Tensor):
            flow2avatar_output[k] = v.detach().cpu()
    with open(osj(args.exp_dir, 'smpld.pkl'), 'wb') as f:
        pickle.dump(flow2avatar_output, f)

    # if args.visualize:
    #     render_runner.render(
    #         renderer=flow2avatar.renderer_rgb,
    #         device=device,
    #         meshes=avatar,
    #         cameras=cameras,
    #         no_grad=True,
    #         return_tensor=False,
    #         output_path=osj(args.exp_dir, 'demo.mp4'),
    #     )


if __name__ == '__main__':
    main()
