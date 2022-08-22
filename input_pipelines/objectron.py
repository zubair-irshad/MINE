import random
import os
import numpy as np
from PIL import Image
from collections import defaultdict

import torch
from torch.utils.data.dataloader import default_collate
import torchvision.transforms as transforms
import torch.utils.data as data
import sys
sys.path.append('/home/zubair/MINE')
from input_pipelines import colmap_utils
import pickle
import glob 
import cv2

def _collate_fn(batch):
    _src_items, _tgt_items = zip(*batch)

    # Gather and stack tgt infos
    tgt_items = defaultdict(list)
    for si in _tgt_items:
        for k, v in si.items():
            tgt_items[k].append(default_collate(v))

    for k in tgt_items.keys():
        tgt_items[k] = torch.stack(tgt_items[k], axis=0)

    src_items = default_collate(_src_items)
    src_items = {k: v for k, v in src_items.items()
                 if k != "G_cam_world"}
    return src_items, tgt_items

class ObjectronMultiDataset(data.Dataset):
    def __init__(self, config, logger, root, is_validation, img_size=(160, 120),
                 supervision_count=5, visible_points_count=100, img_pre_downsample_ratio=12):
        self.config = config,
        self.base_dir = root
        self.logger = logger
        self.is_validation = is_validation
        self.supervision_count = supervision_count
        self.val_instances = [30, 60, 90, 120, 145]
        self._init_img_transforms()
        self.visible_points_count = visible_points_count
        self.crop_transform = transforms.CenterCrop((384,640))
        # self.crop_transform = transforms.CenterCrop((640,384))
        self.render_kwargs = {
            'min_dist': 0.08,
            'max_dist': 0.8}
        self.collate_fn = _collate_fn

        self.adjust_matrix = np.array(
            [[0.,   1.,   0., 0],
            [1.,   0.,   0., 0],
            [0.,   0.,  -1., 0], 
             [0, 0, 0, 1] ])
        self.keys = []
        # Read data
        self.dataset_infos = defaultdict(dict)
        self.scene_to_indices = defaultdict(set)
        index = 0
        self.ids = np.sort([f.name for f in os.scandir(self.base_dir)])
        for i, scene_name in enumerate(np.sort([f.name for f in os.scandir(self.base_dir)])):
            scene_dir = os.path.join(self.base_dir, scene_name)

            if self.is_validation:
                masks_folder = 'masks_3_val'
            else:
                masks_folder = 'masks_3'

            masks_dir = os.path.join(scene_dir, masks_folder+'/*.png')
            allmaskfiles = sorted(glob.glob(masks_dir))
            maskfiles = np.array(allmaskfiles)
            all_imgs =[]
            if self.is_validation:
                image_folder = 'images_3_val'
            else:
                image_folder = 'images_3'

            # instance_dir = os.path.join(self.base_dir, scene_dir)
            meta_data_filename = scene_dir+ '/'+ scene_name+'_metadata.pickle'
            with open(meta_data_filename, 'rb') as handle:
                meta_data = pickle.load(handle)
                
            all_c2w = meta_data["poses"]
            all_focal = meta_data['focal'] 
            axis_align_mat = meta_data['RT'] 
            scale = meta_data['scale']
            scene_points_3d = meta_data['all_scene_points']  
            all_c = meta_data["c"]
            
            for img_id, seg_name in enumerate(maskfiles):
                img_name = os.path.basename(str(seg_name)).split('_')[1]
                img_path = os.path.join(scene_dir, image_folder, img_name)

                self.scene_to_indices[scene_name].add(index)
                self.keys.append((scene_name, img_path))
                focal_idx = int(img_name.split('.')[0])
                index += 1
                # print("focal_idx", focal_idx)
                
                # print("np.array(all_c2w)", np.array(all_c2w).shape)
                c2w = np.array(all_c2w)[focal_idx]
                # c2w = np.linalg.inv(np.squeeze(c2w))
                c2w = np.squeeze(c2w)

                # c2w = c2w @ np.diag([1, -1, -1, 1])
                # c2w = np.linalg.inv(c2w @ np.diag([1, -1, -1, 1]))
                c2w = np.linalg.inv(c2w @ self.adjust_matrix) 

                
                
                # print("scene_points_3d[focal_idx]", scene_points_3d[focal_idx].shape)
                self.dataset_infos[scene_name][img_path] = self._info_transform(
                    {"img_path": img_path, "G_cam_world": c2w, "xyzs": scene_points_3d,
                     "focal": all_focal[focal_idx],
                     "c": all_c[focal_idx]},
                    img_pre_downsample_ratio
                )
                if img_id>149:
                    break
            self.length = len(self.keys)
        self.length = len(self.keys)
        if self.logger:
            self.logger.info("Dataset root: {}, is_validation: {}, number of images: {}"
                             .format(root, self.is_validation, self.length))
                
    def _info_transform(self, info, downsample_ratio):
        img = cv2.imread(info["img_path"])
        img = img[...,::-1]
        img = Image.fromarray(img)
        img = img.transpose(Image.ROTATE_90)
        img = self.img_transforms(img) # (h, w, 3)
        img = self.crop_transform(img)
        _info = {"img": img}
        # print("info[G_cam_world] before", info["G_cam_world"])
        _info["G_cam_world"] = info["G_cam_world"]
        # print("info[G_cam_world] after", _info["G_cam_world"])
        xyzs_world = np.array(info["xyzs"])
        xyzs_world_homo = np.hstack((xyzs_world,
                                     np.ones((len(xyzs_world), 1)))).T.astype(np.float32)

        # Transform xyzs to camera coordiantes
        xyzs_cam_homo = _info["G_cam_world"] @ xyzs_world_homo
        xyzs_cam_homo /= xyzs_cam_homo[-1]

        # Assuming simple_radial camera model
        # Compute K matrix
        f_x = info["focal"][0] 
        f_y = info["focal"][1] 
        p_x = info["c"][0] 
        p_y = info["c"][1] 
        _info["K"] = np.array([
            [f_x, 0, p_x],
            [0, f_y, p_y],
            [0, 0, 1]
        ], dtype=np.float32)
        # _info["K"] = np.array([
        #     [f_x, 0, p_y],
        #     [0, f_y, p_x],
        #     [0, 0, 1]
        # ], dtype=np.float32)
        # print(" _info[K]",  _info["K"])
        # print("_info[img", _info["img"].shape)
        _info["K_inv"] = np.linalg.inv(_info["K"])
        _info["xyzs"] = xyzs_cam_homo[:-1]
        return _info

    def _sample_tgt_items(self, src_idx, src_item):
        G_src_world = src_item["G_cam_world"]
        scene_name, _ = self.keys[src_idx]

        # randomly sample K items for supervision, excluding the src_idx
        # scene_indices = [i for i in self.scene_to_indices[scene_name] if i != src_idx]
        scene_indices = []
        for i in self.scene_to_indices[scene_name]:
            # print("i", i)
            if i == src_idx:
                continue
            if i >src_idx+10:
                continue
            if i < src_idx-10:
                continue
            else:
                scene_indices.append(i)
        if not self.is_validation:
            sampled_indices = random.sample(scene_indices, self.supervision_count)
        else:
            sampled_indices = [scene_indices[(src_idx + 1) % (len(scene_indices)) - 1]]

        print("target sampled_indices", sampled_indices)
        # Generate sampled_items and calculate the relative rotation matrix and translation vector
        # accordingly.
        sampled_items = defaultdict(list)
        for index in sampled_indices:
            _, img_path = self.keys[index]
            img_info = self.dataset_infos[scene_name][img_path]

            G_tgt_world = img_info["G_cam_world"]
            G_src_tgt = G_src_world @ np.linalg.inv(G_tgt_world)

            # G_src_tgt = np.linalg.inv(G_src_tgt)

            sampled_items["img"].append(img_info["img"])
            sampled_items["K"].append(img_info["K"])
            sampled_items["K_inv"].append(img_info["K_inv"])
            sampled_items["G_src_tgt"].append(G_src_tgt)

            # Sample xyz points
            # TODO: deterministic behavior in val
            # sampled_xyzs_indices = random.sample(range(len(img_info["xyzs_ids"])),
            #                                      self.visible_points_count) \
            #     if not self.is_validation \
            #     else sorted(range(len(img_info["xyzs_ids"]))[:256])
            # print("img_info[xyzs]", img_info["xyzs"].shape)
            sampled_xyzs_indices = random.sample(range(img_info["xyzs"].shape[1]),
                                                 self.visible_points_count)
            sampled_items["xyzs"].append(img_info["xyzs"][:, sampled_xyzs_indices])
        return sampled_items

    def __len__(self):
        return self.length

    def _init_img_transforms(self):
        self.img_transforms = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        # Read src item
        # print("length", self.length)
        print("index", index)
        scene_name, img_path = self.keys[index]

        print("scene_name, img_path", scene_name, img_path)
        # print("scene_name, img_path", scene_name, img_path)
        _src_item = self.dataset_infos[scene_name][img_path]

        # Copy new src_item
        src_item = {k: v for k, v in _src_item.items()}

        tgt_items = self._sample_tgt_items(index, src_item)
        # Sample 3D points in src items
        # TODO: deterministic behavior in val
        # print("_src_item[xyzs]", _src_item["xyzs"].shape)
        sampled_indices = random.sample(range(_src_item["xyzs"].shape[1]),
                                        self.visible_points_count)
        src_item["xyzs"] = src_item["xyzs"][:, sampled_indices]
        return src_item, tgt_items


if __name__ == "__main__":
    import logging
    dataset = ObjectronMultiDataset({},logger =logging,
                          root="/home/zubair/MINE/camera",
                          is_validation=False,
                          img_size=(160, 120),
                          supervision_count=1,
                          img_pre_downsample_ratio=12)
    from torch.utils.data import DataLoader

    dl = DataLoader(dataset, batch_size=4, shuffle=False,
                    drop_last=True, num_workers=0,
                    collate_fn=_collate_fn)

    for batch in dl:
        src_item, supervision_items = batch

        # for k, v in src_item.items():
        #     print(k, v.size())

        # print("=========================\n\n\n")

        # for k, v in supervision_items.items():
        #     print(k, v.size())
            # if k=='img':
            #     print("imgsssss", v.shape)
            # if k=='xyzs':
            #     print("pointsssssssss", v.shape)
            #     points = v

        # for i in range(points.shape[0]):
        #     points_np = v[i, :,:].permute(1,0).numpy()
        #     print("points_np", points_np.shape)
        #     pcd = o3d.geometry.PointCloud()
        #     pcd.points = o3d.utility.Vector3dVector(points_np)
        #     o3d.visualization.draw_geometries([pcd])
        # print("********")

        # for k, v in supervision_items.items():
        #     print(k, v.size())

        break