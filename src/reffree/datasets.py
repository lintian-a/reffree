import torch
from torch.utils.data import Dataset
import glob
import os
import pickle
import torch.nn.functional as F
import nibabel as nib
import numpy as np
from torch.nn.functional import conv3d, conv2d
from monai.transforms import ResizeWithPadOrCropd

synth_config = {
    "max_rotation":15,
    "max_shear":0.2,
    "max_scaling":0.2,
    "crop_max_translation":0.25,
    "crop_scale":0.75,
    "nonlin_scale_min":0.03,
    "nonlin_scale_max":0.06,
    "nonlin_std_max":4,
    "bf_scale_min":0.02,
    "bf_scale_max":0.04,
    "bf_std_min":0.1,
    "bf_std_max":0.6,
    "gamma_std":0.1,
    "quantile_normalization":0.99,
}

class MRIHemiDataset(Dataset):
    def __init__(self, data_folder, split_folder="", cm_file=None, use_split=False, phase="train", dtype=torch.float32, shape=(196, 256, 256)):  
        # Load the data list according to the phase
        names = []
        if phase == "train" or phase == "debug":
            # When phase is debug, we load the training dataset and only use the first 10 scans. 
            # The logic of getting the first 10 scanes in debug phase locates at several lines after.
            if use_split:
                with open(f'{split_folder}/train_ids.pkl', "rb") as f:
                    names = pickle.load(f)
            else:
                # Use all data
                # Collect list of available images, per dataset
                datasets = []
                g = glob.glob(os.path.join(data_folder, '*' + 'T1w.nii'))
                for i in range(len(g)):
                    filename = os.path.basename(g[i])
                    dataset = filename[:filename.find('.')]
                    found = False
                    for d in datasets:
                        if dataset==d:
                            found = True
                    if found is False:
                        datasets.append(dataset)
                for i in range(len(datasets)):
                    names.append(glob.glob(os.path.join(data_folder, datasets[i] + '.*' + 'T1w.nii')))
        elif phase == "val":
            with open(f'{split_folder}/val_ids.pkl', "rb") as f:
                names = pickle.load(f)
        
        # Create a weighted sampler according to the number of images in each dataset
        weights = []
        for i in range(len(names)):
            weights.append(torch.ones(len(names[i]))* 1./len(names[i]))
        weights = torch.cat(weights) * 1./len(names)
        self.sampler = torch.utils.data.sampler.WeightedRandomSampler(
                    weights=weights,
                    num_samples=len(weights),
                    replacement=False)

        # Change the by-dataset list to by-case list
        self.names = []
        for i in range(len(names)):
            self.names.extend(names[i])
        
        # Load the center of mass file
        if cm_file is not None:
            with open(cm_file, "rb") as f:
                self.center_of_mass_dict = pickle.load(f)
        else:
            self.center_of_mass_dict = None

        # If phase is debug, we only use the first 10 scans
        if phase == "debug":
            self.names = self.names[:10]
        print('Use ' + str(len(self.names)) + ' scans in total for ' + phase)

        self.dtype = dtype
        self.resize = ResizeWithPadOrCropd(["mni"], shape, mode="edge")
        self.resize_zero = ResizeWithPadOrCropd(["roi", "gen_labels"], shape, mode="constant", constant_values=0)

        # prepare look up tables we will use to binarize segmentation mask and extract a single hemisphere
        # self.lut[0] represents labels on left hemisphere, self.lut[1] represents labels on right hemisphere
        self.lut = torch.zeros((2, 100), dtype=torch.long)
        self.lut[0][[1,2,3,4,7,8,9,10,14,15,17,34,36,38,40,42]] = 1
        self.lut[1][[18,19,20,21,24,25,26,27,28,29,30,35,37,39,41,43]] = 1

        self.half_shape = (shape[0]//2, shape[1]//2, shape[2]//2)
    
    def __getitem__(self, index):
        name = self.names[index][:-7]
        side = 'left'
        if (np.random.rand() < 0.5):
            side = 'right'
        center_of_mass = self.center_of_mass_dict[side+'_cm'][name] if self.center_of_mass_dict is not None else None

        roi, gen_labels, mni = self.load_imgs(
            name, side, center_of_mass, dtype=self.dtype)
        
        data_with_zero_padding = self.resize_zero(
            {
                "roi": roi,
                "gen_labels": gen_labels,
                
            }
        )
        data_with_edge_padding = self.resize(
            {
                "mni": mni
            }
        )
        
        return (
            data_with_zero_padding["roi"], 
            data_with_zero_padding["gen_labels"], 
            data_with_edge_padding["mni"])
    
    def __len__(self):
        return len(self.names)
    
    def load_imgs(self, name, side, center_of_mass, dtype=torch.float32):
        '''
        Load images from the dataset
        
        Returns:
        segmentation_labels: torch.tensor, segmentation labels. 1x1xHxWxD, int
        generation_labels: torch.tensor, generation labels. 1x1xHxWxD, int
        pial_distances: torch.tensor, pial distances. 1x1xHxWxD, int
        mni: torch.tensor, MNI coordinates. 1x3xHxWxD, dtype
        '''
        # files we'll need to read
        generation_labels = name + 'generation_labels.nii'
        segmentation_labels = name + 'brainseg.nii'
        pial_distances = name + side[0] + 'p_dist_map.nii'
        mni_reg_x = name + 'mni_reg.x.nii'
        mni_reg_y = name + 'mni_reg.y.nii'
        mni_reg_z = name + 'mni_reg.z.nii'

        def get_img(file_name, bbox, dtype):
            if bbox:
                img = torch.squeeze(torch.tensor(
                    nib.load(file_name).slicer[
                        bbox[0][0]:bbox[0][1], 
                        bbox[1][0]:bbox[1][1], 
                        bbox[2][0]:bbox[2][1]
                        ].get_fdata(), dtype=dtype))
            else:
                img = torch.squeeze(torch.tensor(
                    nib.load(file_name).get_fdata(), dtype=dtype
                ))
            return img

        # Peek the shape of the image
        seg = nib.load(segmentation_labels).get_fdata()
        bbox = [[int(center_of_mass[0])-self.half_shape[0], int(center_of_mass[0])+self.half_shape[0]],
                 [int(center_of_mass[1])-self.half_shape[1], int(center_of_mass[1])+self.half_shape[1]],
                 [int(center_of_mass[2])-self.half_shape[2], int(center_of_mass[2])+self.half_shape[2]]]

        for i in range(3):
            if bbox[i][0] < 0:
                bbox[i][0] = 0
            if bbox[i][1] > seg.shape[i]:
                bbox[i][1] = seg.shape[i]

        # Load data
        segmentation_labels = get_img(segmentation_labels, bbox, torch.int)[None]
        generation_labels = get_img(generation_labels, bbox, torch.int)[None]
        pial_distances = get_img(pial_distances, bbox, torch.int)[None]
        MNIx = 0.0001 * get_img(mni_reg_x, bbox, dtype)
        MNIy = 0.0001 * get_img(mni_reg_y, bbox, dtype)
        MNIz = 0.0001 * get_img(mni_reg_z, bbox, dtype)
        mni = torch.stack((MNIx, MNIy, MNIz), dim=0)

        roi = self.lut[int(side=='right')][segmentation_labels] | ((segmentation_labels == 31) & (pial_distances < 128))

        if side == "right":
            mni = torch.flip(mni, [1])
            mni[0,:,:,:] = -mni[0,:,:,:]
            generation_labels = torch.flip(generation_labels, [1])
            roi = torch.flip(roi, [1])

        return roi, generation_labels, mni

class PhotoSynthReconTaskGenerator:
    def __init__(self, in_shape, slice_shape, slices_per_volume, synth_config=synth_config, device='cuda'):

        self.in_shape = in_shape
        self.shape = (slice_shape[0], in_shape[1], slice_shape[1])
        self.dtype = torch.float32
        self.device = device
        self.synth_config = synth_config
        self.slices_per_volume = slices_per_volume  
        self.min_roi_area_for_training = 25

        self.identity = F.affine_grid(torch.eye(4)[:3].unsqueeze(0), (1, 1,) + self.shape, align_corners=True).to(device)
        self.index = F.affine_grid(torch.eye(4)[:3].unsqueeze(0), (1, 1,) + self.in_shape, align_corners=True).to(device).permute(0,4,1,2,3)

    def get_task_data(self, roi, gen_labels, mni):
        batch_size = roi.shape[0]

        roi_center = ((self.index * roi).sum(dim=(2,3,4))/roi.sum(dim=(2,3,4))).flip(1)
        
        # # Synthesize deformations
        synth_deform = self.synth_deform(batch_size, self.dtype, self.device)
        synth_affine = torch.vmap(self.synth_affine, randomness="different")(roi_center)

        # Simulate random crop
        synth_trans = (torch.rand((batch_size, 1, roi.shape[3], 1, 3), device=self.device) * 2.0 - 1.0) * self.synth_config["crop_max_translation"]
        synth_trans[:,:,:,:,1] = 0
        synth_scale = torch.ones_like(synth_trans, device=self.device) * self.synth_config["crop_scale"]
        synth_scale[:,:,:,:,1] = 1

        # # Get slice numbers. The slice number should be computed after affine transformation.
        composed = synth_affine[:,None,None,None] @ torch.concat(
            [
                (self.identity).expand(batch_size, -1, -1, -1, -1).flip(-1), 
                torch.ones(list(synth_deform.shape[:-1])+[1,], device=self.device)
                ], dim=-1)[:,:,:,:,:,None]
        composed = composed.squeeze(-1).flip(-1)
        roi_affine = F.grid_sample(roi.float(), composed, mode='bilinear', align_corners=True)
        roi_affine[roi_affine < 0.5] = 0
        roi_affine[roi_affine >= 0.5] = 1
        valid_slices = torch.where(roi_affine.sum(dim=(2,4))>0, 1, 0)
        min_j = torch.argmax(valid_slices, dim=2) + 1
        max_j = valid_slices.shape[2] - 1 - torch.argmax(valid_slices.flip(2), dim=2) - 1
        slice_numbers = (torch.arange(roi.shape[3], device=self.device)[None].expand(roi.shape[0], -1) - (min_j-1)) / (max_j-min_j+1)
        slice_numbers[slice_numbers < 0] = 0
        slice_numbers[slice_numbers > 1] = 1

        # sample random slices to deform. 
        # The selected slices should have enough valid voxels (e.g., 4) after deformation.
        composed = synth_affine[:,None,None,None] @ torch.concat([(synth_deform + self.identity * synth_scale + synth_trans).flip(-1), torch.ones(list(synth_deform.shape[:-1])+[1,], device=self.device)], dim=-1)[:,:,:,:,:,None]
        # composed = synth_affine[:,None,None,None] @ torch.concat([(synth_deform + self.identity).flip(-1), torch.ones(list(synth_deform.shape[:-1])+[1,], device=self.device)], dim=-1)[:,:,:,:,:,None]
        composed = composed.squeeze(-1).flip(-1)
        roi = F.grid_sample(roi.float(), composed, mode='bilinear', align_corners=True)
        roi[roi <= 0.5] = 0
        roi[roi > 0.5] = 1
        valid_slices = torch.where(roi.sum(dim=(2,4))>self.min_roi_area_for_training, 1, 0)
        valid_slices = valid_slices.cpu()
        # composed, roi, slice_numbers = torch.vmap(self.select_slices, randomness="different")(valid_slices, composed, roi, slice_numbers)
        
        # TODO: Parallelize this
        composed_selected = []
        roi_selected = []
        slice_numbers_selected = []
        for i in range(valid_slices.shape[0]):
            valid_idx = torch.where(valid_slices[i,0]>0)[0]
            selected = valid_idx[torch.randperm(len(valid_idx))][:self.slices_per_volume]
            composed_selected.append(composed[i:i+1,:,selected])
            roi_selected.append(roi[i:i+1,:,:,selected])
            slice_numbers_selected.append(slice_numbers[i:i+1,selected])
        composed = torch.concat(composed_selected, dim=0)
        roi = torch.concat(roi_selected, dim=0)
        slice_numbers = torch.concat(slice_numbers_selected, dim=0)

        # # Apply deformation to slices
        gen_labels = F.grid_sample(gen_labels.float(), composed, mode='nearest', align_corners=True)
        mni = F.grid_sample(mni, composed, mode='bilinear', align_corners=True)

        # # Augment intensity. Change image from 1x1xHxWxD to 1xWxHxD. Then we augment it as a list of 2D images.
        imgs = torch.vmap(self.aug_contrast, in_dims=0, randomness='different')(
            gen_labels.permute(0,3,1,2,4).reshape(-1, self.shape[0], self.shape[2])).reshape(batch_size, -1, 1, self.shape[0], self.shape[2]).permute(0,2,3,1,4)

        imgs = self.aug_intensity(imgs, roi)


        # Mask the mni
        mni = mni * roi
        
        # Reshape to be list of 2D images
        imgs = imgs.permute(0,3,1,2,4).reshape(-1, 1, self.shape[0], self.shape[2])
        roi = roi.permute(0,3,1,2,4).reshape(-1, 1, self.shape[0], self.shape[2])
        mni = mni.permute(0,3,1,2,4).reshape(-1, 3, self.shape[0], self.shape[2])
        slice_numbers = slice_numbers.reshape(-1)
        return imgs, roi, mni, slice_numbers

    # def select_slices(self, valid_slices, composed, roi, slice_numbers):
    #     valid_idx = torch.where(valid_slices>0)[0]
    #     selected = valid_idx[torch.randperm(len(valid_idx))][:self.slices_per_volume]
    #     return composed[:, :, selected], roi[:, :, selected], slice_numbers[selected]

    def synth_deform(self, batch_size, dtype, device):
        '''
        Synthesize deformation field

        Returns:
        disp: torch.tensor, deformation field. batch_sizexHxWxDx3, dtype
        '''
        # sample nonlinear deformation, letting every slice do its own thing stuff
        nonlin_scale = self.synth_config["nonlin_scale_min"] + np.random.rand(1) * (self.synth_config["nonlin_scale_max"] - self.synth_config["nonlin_scale_min"])
        siz_F_small = np.round(nonlin_scale * np.array(self.shape)).astype(int).tolist()
        siz_F_small[1] = self.shape[1] # we actually go crazy in every slice
        nonlin_std = self.synth_config["nonlin_std_max"] * np.random.rand() / np.array(self.shape) * 2.0 # nonlin_std_max is defined in voxels. We want it in canonical space [-1, 1].
        Fsmall = torch.from_numpy(nonlin_std).float().to(device)[None,None,None] * torch.randn([batch_size, *siz_F_small, 3], dtype=dtype, device=device)
        disp = torch.nn.functional.interpolate(Fsmall.permute(0,4,1,2,3), size=self.shape, mode='trilinear', align_corners=False).permute(0,2,3,4,1)
        disp[:, :, :, :, 1] = 0 # no deformation across slices
        return disp
    
    def synth_affine(self, roi_center):
        '''
        Synthesize affine matrix
        
        Returns:
        A: torch.tensor, affine deformation. 3x4, dtype'''
        # sample affine deformation A and center c2
        rotations = (2 * self.synth_config["max_rotation"] * np.random.rand(3) - self.synth_config["max_rotation"]) / 180.0 * np.pi
        shears = (2 * self.synth_config["max_shear"] * np.random.rand(3) - self.synth_config["max_shear"])
        scalings = 1 + (2.0 * np.random.rand(3) - 1.0) * self.synth_config["max_scaling"]
        A = torch.tensor(make_affine_matrix(rotations, shears, scalings), dtype=torch.float, device=self.device)
        A = torch.concat([A, roi_center[:,None]], dim=1)

        # TODO: debug
        # A = torch.eye(3).to(self.device)
        # A = torch.concat([A, roi_center[:,None]], dim=1)
        return A
    
    def compose(self, phi, psi):
        '''
        Compose two deformation fields
        
        Args:
        phi: torch.tensor, deformation field represented in pytorch grids. 1xHxWxDx3
        psi: torch.tensor, deformation field represented in pytorch grids. 1xHxWxDx3'''

        return F.grid_sample(phi.permute(0,4,1,2,3), psi, mode='bilinear', padding_mode='border', align_corners=True).permute(0,2,3,4,1)
    
    def aug_contrast(self, gen_labels):
        '''
        Augment contrast of the image

        Args:
        gen_labels: torch.tensor, generation labels. HxW, int
        '''
        dtype = self.dtype
        device = self.device
        # sample Gaussian parameters
        mus = 25 + 200 * torch.rand(256, dtype=dtype, device=device)
        mus[0] = 0 # background always zero (in case)
        sigmas = 5 + 20 * torch.rand(256, dtype=dtype, device=device)
        # Partial volume bit: 1 = lesion, 2 = WM, 3 = GM, 4 = CSF
        v = 0.02 * torch.arange(50).to(device)
        mus[100:150] = mus[1] * (1 - v) + mus[2] * v
        mus[150:200] = mus[2] * (1 - v) + mus[3] * v
        mus[200:250] = mus[3] * (1 - v) + mus[4] * v
        mus[250] = mus[4]
        sigmas[100:150] = torch.sqrt(sigmas[1] ** 2 * (1 - v) + sigmas[2] ** 2 * v)
        sigmas[150:200] = torch.sqrt(sigmas[2] ** 2 * (1 - v) + sigmas[3] ** 2 * v)
        sigmas[200:250] = torch.sqrt(sigmas[3] ** 2 * (1 - v) + sigmas[4] ** 2 * v)
        sigmas[250] = sigmas[4]

        # sample Gaussian image
        gen_labels = gen_labels.int()
        syn_img = mus[gen_labels] + sigmas[gen_labels] * torch.randn(gen_labels.shape, dtype=dtype, device=device)
        return syn_img

    def aug_intensity(self, img, mask):
        '''
        Augment intensity of the image

        Args:
        img: torch.tensor, image. Bx1xHxWxD
        '''
        # OK now some further intensity augmentation!
        # cosmetic blurring
        IMAGEblur = gaussian_blur_3d(img, np.array([.5, 0., .5]), self.device, dtype=self.dtype)
        # uneven illumination field
        bf_scale = self.synth_config["bf_scale_min"] + np.random.rand(1) * (self.synth_config["bf_scale_max"] - self.synth_config["bf_scale_min"])
        siz_BF_small = np.round(bf_scale * np.array(img.shape[2:])).astype(int).tolist()
        siz_BF_small = [1, 1] + siz_BF_small
        siz_BF_small[3] = img.shape[3]
        BFsmall = torch.tensor(self.synth_config["bf_std_min"] + (self.synth_config["bf_std_max"] - self.synth_config["bf_std_min"]) * np.random.rand(1), dtype=self.dtype,
                                device=self.device) * torch.randn(siz_BF_small, dtype=self.dtype, device=self.device)
        BFlog = F.interpolate(BFsmall, size=img.shape[2:], mode='trilinear', align_corners=False)
        BF = torch.exp(BFlog)
        IMAGEbf = IMAGEblur * BF

        # Normalize to 0,1 and mask
        MASKout = torch.logical_not(mask)
        IMAGEbf[IMAGEbf<0] = 0
        IMAGEbf[MASKout] = 0
        mask = mask.bool()

        batch_size = img.shape[0]
        for b in range(batch_size):
            for s in range(img.shape[3]):   
                IMAGEbf[b, :, :, s,:] /= torch.quantile(IMAGEbf[b, :, :, s,:][mask[b, :, :, s,:]], self.synth_config["quantile_normalization"])
        IMAGEbf[IMAGEbf>1] = 1
        # gamma transform
        gamma = torch.exp(self.synth_config["gamma_std"] * torch.randn((batch_size,1,1,img.shape[3],1), device=self.device))
        IMAGEbf = IMAGEbf ** gamma
        return IMAGEbf


def make_affine_matrix(rot, sh, s):
    Rx = np.array([[1, 0, 0], [0, np.cos(rot[0]), -np.sin(rot[0])], [0, np.sin(rot[0]), np.cos(rot[0])]])
    Ry = np.array([[np.cos(rot[1]), 0, np.sin(rot[1])], [0, 1, 0], [-np.sin(rot[1]), 0, np.cos(rot[1])]])
    Rz = np.array([[np.cos(rot[2]), -np.sin(rot[2]), 0], [np.sin(rot[2]), np.cos(rot[2]), 0], [0, 0, 1]])

    SHx = np.array([[1, 0, 0], [sh[1], 1, 0], [sh[2], 0, 1]])
    SHy = np.array([[1, sh[0], 0], [0, 1, 0], [0, sh[2], 1]])
    SHz = np.array([[1, 0, sh[0]], [0, 1, sh[1]], [0, 0, 1]])

    A = SHx @ SHy @ SHz @ Rx @ Ry @ Rz
    A[0, :] = A[0, :] * s[0]
    A[1, :] = A[1, :] * s[1]
    A[2, :] = A[2, :] * s[2]

    return A

def make_gaussian_kernel(sigma, device, dtype):
    sl = int(np.ceil(3 * sigma))
    ts = torch.linspace(-sl, sl, 2 * sl + 1, dtype=dtype, device=device)
    gauss = torch.exp((-(ts / sigma) ** 2 / 2))
    kernel = gauss / gauss.sum()

    return kernel

def gaussian_blur_2d(input, stds, device, dtype=torch.float):
    blurred = input
    if stds[0] > 0:
        kx = make_gaussian_kernel(stds[0], device=device, dtype=dtype)
        blurred = conv2d(blurred, kx[None, None, :, None], stride=1, padding=(len(kx) // 2, 0))
    if stds[1] > 0:
        ky = make_gaussian_kernel(stds[1], device=device, dtype=dtype)
        blurred = conv2d(blurred, ky[None, None, None, :], stride=1, padding=(0, len(ky) // 2))
    return blurred

def gaussian_blur_3d(input, stds, device, dtype=torch.float):
    blurred = input
    if stds[0] > 0:
        kx = make_gaussian_kernel(stds[0], device=device, dtype=dtype)
        blurred = conv3d(blurred, kx[None, None, :, None, None], stride=1, padding=(len(kx) // 2, 0, 0))
    if stds[1] > 0:
        ky = make_gaussian_kernel(stds[1], device=device, dtype=dtype)
        blurred = conv3d(blurred, ky[None, None, None, :, None], stride=1, padding=(0, len(ky) // 2, 0))
    if stds[2] > 0:
        kz = make_gaussian_kernel(stds[2], device=device, dtype=dtype)
        blurred = conv3d(blurred, kz[None, None, None, None, :], stride=1, padding=(0, 0, len(kz) // 2))
    return blurred