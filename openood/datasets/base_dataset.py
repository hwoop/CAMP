from torch.utils.data import Dataset


class BaseDataset(Dataset):

    def __getitem__(
        self,
        index,
    ):

        # in some pytorch versions, input index will be torch.Tensor
        index = int(index)
        sample = self.getitem(index)
        sample['index'] = index
        return sample


    def getitem(
        self,
        index,
    ):
        raise NotImplementedError
