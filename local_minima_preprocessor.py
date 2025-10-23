import numpy as np
import math
from typing import Tuple, Union
from scipy.ndimage import minimum_filter
from skimage.measure import label

import nnunetv2
from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero
from nnunetv2.preprocessing.resampling.default_resampling import compute_new_shape
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager


class LocalMinimaPreprocessor(DefaultPreprocessor):
    @staticmethod
    def _sample_local_minima(data, classes_or_regions, properties, dataset_json, seed=1234, verbose=False):
        num_samples = 1e7
        min_percent_coverage = 0.01  # at least 1% of the class voxels need to be selected, otherwise it may be too
                                     # sparse
        rndst = np.random.RandomState(seed)
        class_locs = {}

        spacing = properties['spacing']
        data_index = next(int(idx) for idx, mod in dataset_json['channel_names'].items() if mod == 'T2S')
        if verbose:
            print(f"_sample_local_minima() data shape: {data.shape}")
            print(f"_sample_local_minima() spacing: {spacing}")
        footprint_shape = tuple([int(10 // s) for s in spacing])
        if verbose:
            print(f"_sample_local_minima() footprint_shape: {footprint_shape}")
        footprint = np.ones(footprint_shape)
        data = data[data_index, ...]
        min_filtered = minimum_filter(data, footprint=footprint, mode='constant')
        minima = np.zeros_like(data)
        minima[np.nonzero(data == min_filtered)] = 1
        minima_labels = label(minima)
        # Discard corners
        for z in (0, data.shape[0] - 1):
            for y in (0, data.shape[1] - 1):
                for x in (0, data.shape[2] - 1):
                    l = minima_labels[z, y, x]
                    if l == 0:
                        continue
                    mask = minima_labels == l
                    minima_labels[mask] = 0
                    minima[mask] = 0
        minima = np.expand_dims(minima, axis=0)
        foreground_coords = np.argwhere(minima == 1)
        
        # We don't need more than 1e7 foreground samples. That's insanity. Cap here
        if len(foreground_coords) > 1e7:
            take_every = math.floor(len(foreground_coords) / 1e7)
            # keep computation time reasonable
            if verbose:
                print(f'Subsampling foreground pixels 1:{take_every} for computational reasons')
            foreground_coords = foreground_coords[::take_every]
        
        if verbose:
            print(f"_sample_local_minima() found {len(foreground_coords)} coords")
        assert len(classes_or_regions) == 1, f"LocalMinimaPreprocessor currently supports only binary segmentation whereas the given data has {classes_or_regions} classes"
        for c in classes_or_regions:
            k = c if not isinstance(c, list) else tuple(c)
            all_locs = foreground_coords
            if len(all_locs) == 0:
                class_locs[k] = []
                continue
            target_num_samples = min(num_samples, len(all_locs))
            target_num_samples = max(target_num_samples, int(np.ceil(len(all_locs) * min_percent_coverage)))
            selected = all_locs[rndst.choice(len(all_locs), target_num_samples, replace=False)]
            class_locs[k] = selected
        return class_locs

    def run_case_npy(self, data: np.ndarray, seg: Union[np.ndarray, None], properties: dict,
                     plans_manager: PlansManager, configuration_manager: ConfigurationManager,
                     dataset_json: Union[dict, str]):
        # let's not mess up the inputs!
        data = data.astype(np.float32)  # this creates a copy
        if seg is not None:
            assert data.shape[1:] == seg.shape[1:], "Shape mismatch between image and segmentation. Please fix your dataset and make use of the --verify_dataset_integrity flag to ensure everything is correct"
            seg = np.copy(seg)

        has_seg = seg is not None

        # apply transpose_forward, this also needs to be applied to the spacing!
        data = data.transpose([0, *[i + 1 for i in plans_manager.transpose_forward]])
        if seg is not None:
            seg = seg.transpose([0, *[i + 1 for i in plans_manager.transpose_forward]])
        original_spacing = [properties['spacing'][i] for i in plans_manager.transpose_forward]

        # crop, remember to store size before cropping!
        shape_before_cropping = data.shape[1:]
        properties['shape_before_cropping'] = shape_before_cropping
        # this command will generate a segmentation. This is important because of the nonzero mask which we may need
        data, seg, bbox = crop_to_nonzero(data, seg)
        properties['bbox_used_for_cropping'] = bbox
        # print(data.shape, seg.shape)
        properties['shape_after_cropping_and_before_resampling'] = data.shape[1:]

        # resample
        target_spacing = configuration_manager.spacing  # this should already be transposed

        if len(target_spacing) < len(data.shape[1:]):
            # target spacing for 2d has 2 entries but the data and original_spacing have three because everything is 3d
            # in 2d configuration we do not change the spacing between slices
            target_spacing = [original_spacing[0]] + target_spacing
        new_shape = compute_new_shape(data.shape[1:], original_spacing, target_spacing)

        # normalize
        # normalization MUST happen before resampling or we get huge problems with resampled nonzero masks no
        # longer fitting the images perfectly!
        data = self._normalize(data, seg, configuration_manager,
                               plans_manager.foreground_intensity_properties_per_channel)

        # print('current shape', data.shape[1:], 'current_spacing', original_spacing,
        #       '\ntarget shape', new_shape, 'target_spacing', target_spacing)
        old_shape = data.shape[1:]
        data = configuration_manager.resampling_fn_data(data, new_shape, original_spacing, target_spacing)
        seg = configuration_manager.resampling_fn_seg(seg, new_shape, original_spacing, target_spacing)
        if self.verbose:
            print(f'old shape: {old_shape}, new_shape: {new_shape}, old_spacing: {original_spacing}, '
                  f'new_spacing: {target_spacing}, fn_data: {configuration_manager.resampling_fn_data}')

        # if we have a segmentation, sample foreground locations for oversampling and add those to properties
        if has_seg:
            # reinstantiating LabelManager for each case is not ideal. We could replace the dataset_json argument
            # with a LabelManager Instance in this function because that's all its used for. Dunno what's better.
            # LabelManager is pretty light computation-wise.
            label_manager = plans_manager.get_label_manager(dataset_json)
            collect_for_this = label_manager.foreground_regions if label_manager.has_regions \
                else label_manager.foreground_labels

            # when using the ignore label we want to sample only from annotated regions. Therefore we also need to
            # collect samples uniformly from all classes (incl background)
            if label_manager.has_ignore_label:
                collect_for_this.append([-1] + label_manager.all_labels)

            # no need to filter background in regions because it is already filtered in handle_labels
            # print(all_labels, regions)
            # properties['class_locations'] = self._sample_foreground_locations(seg, collect_for_this,
            #                                                                        verbose=self.verbose)
            properties['class_locations'] = self._sample_local_minima(data, collect_for_this, properties, dataset_json, verbose=self.verbose)
        seg = self.modify_seg_fn(seg, plans_manager, dataset_json, configuration_manager)
        if np.max(seg) > 127:
            seg = seg.astype(np.int16)
        else:
            seg = seg.astype(np.int8)
        return data, seg, properties
