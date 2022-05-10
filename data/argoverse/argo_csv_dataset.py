import glob
import pickle

import torch.utils.data

from data.argoverse.utils.extractor_proc import ArgoDataExtractor


class ArgoCSVDataset(torch.utils.data.Dataset):
    def __init__(self, input_folder, input_preprocessed, args):
        self.input_preprocessed = input_preprocessed
        self.args = args

        if args.use_preprocessed:
            with open(input_preprocessed, 'rb') as f:
                self.data = pickle.load(f)
        else:
            self.files = sorted(glob.glob(f"{input_folder}/*.csv"))
            if args.reduce_dataset_size > 0:
                self.files = self.files[:args.reduce_dataset_size]

            self.argo_reader = ArgoDataExtractor(args)

    def __getitem__(self, idx):
        """Get preprocessed data at idx or preprocess the data at idx

        Args:
            idx: Index of the sample to return

        Returns:
            Preprocessed sample
        """
        if self.args.use_preprocessed:
            return self.data[idx]
        else:
            return self.argo_reader.extract_data(self.files[idx])

    def __len__(self):
        """Get number of samples in the dataset

        Returns:
            Number of samples in the dataset
        """
        if self.args.use_preprocessed:
            return len(self.data)
        else:
            return len(self.files)
