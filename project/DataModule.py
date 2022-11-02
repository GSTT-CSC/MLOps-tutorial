from typing import List, Optional

import pytorch_lightning
from mlops.data.tools.tools import xnat_build_dataset
from mlops.data.transforms.LoadImageXNATd import LoadImageXNATd
from monai.data import CacheDataset, pad_list_data_collate
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    LoadImage,
    CropForegroundd,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
)
from torch.cuda import is_available
from torch.utils.data import random_split, DataLoader
from xnat.mixin import ImageScanData, SubjectData


class DataModule(pytorch_lightning.LightningDataModule):

    def __init__(self, data_dir: str = './', xnat_configuration: dict = None, batch_size: int = 1, num_workers: int = 4,
                 test_fraction: float = 0.1, train_val_ratio: float = 0.2, test_batch: int = -1):

        super().__init__()
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.train_val_ratio = train_val_ratio
        self.test_fraction = test_fraction
        self.xnat_configuration = xnat_configuration
        self.test_batch = test_batch

    def setup(self, stage: Optional[str] = None):
        """
        Use the setup method to setup your data and define your Dataset objects
        :param stage:
        :return:
        """
        # list of tuples defining action functions and their data keys
        actions = [(self.fetch_image, 'image'),
                   (self.fetch_label, 'label')]

        self.xnat_data_list = xnat_build_dataset(self.xnat_configuration, actions=actions, test_batch=self.test_batch)

        self.train_samples, self.valid_samples = random_split(self.xnat_data_list, [1-self.train_val_ratio, self.train_val_ratio])

        self.transforms = Compose(
            [
                LoadImageXNATd(keys=['data'], xnat_configuration=self.xnat_configuration,
                               image_loader=LoadImage(image_only=True), expected_filetype_ext='.nii.gz'),
                EnsureChannelFirstd(keys=["image", "label"]),
                # Orientationd(keys=["image", "label"], axcodes="RAS"),
                # # ScaleIntensityRanged(
                # #     keys=["image"], a_min=-57, a_max=164,
                # #     b_min=0.0, b_max=1.0, clip=True,
                # # ),
                # # CropForegroundd(keys=["image", "label"], source_key="image"),
            ]
        )

        self.train_dataset = CacheDataset(data=self.train_samples, transform=self.transforms)
        self.val_dataset = CacheDataset(data=self.valid_samples, transform=self.transforms)

    def prepare_data(self, *args, **kwargs):
        pass

    def train_dataloader(self):
        """
        Define train dataloader
        :return:
        """
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, collate_fn=pad_list_data_collate,
                          pin_memory=is_available())

    def val_dataloader(self):
        """
        Define validation dataloader
        :return:
        """
        return DataLoader(self.val_dataset, batch_size=1, num_workers=self.num_workers, collate_fn=pad_list_data_collate,
                          pin_memory=is_available())


    @staticmethod
    def fetch_image(subject_data: SubjectData = None) -> List[ImageScanData]:
        """
        Function that identifies and returns the required xnat ImageData object from a xnat SubjectData object
        along with the 'key' that it will be used to access it.
        """
        output = []
        for exp in subject_data.experiments:
            for scan in subject_data.experiments[exp].scans:
                if 'image' in subject_data.experiments[exp].scans[scan].id.lower():
                    output.append(subject_data.experiments[exp].scans[scan])
        if len(output) > 1:
            raise TypeError
        return output

    @staticmethod
    def fetch_label(subject_data: SubjectData = None) -> List[ImageScanData]:
        """
        Function that identifies and returns the required xnat ImageData object from a xnat SubjectData object
        along with the 'key' that it will be used to access it.
        """
        output = []
        for exp in subject_data.experiments:
            for scan in subject_data.experiments[exp].scans:
                if 'label' in subject_data.experiments[exp].scans[scan].id.lower():
                    output.append(subject_data.experiments[exp].scans[scan])
        if len(output) > 1:
            raise TypeError
        return output
