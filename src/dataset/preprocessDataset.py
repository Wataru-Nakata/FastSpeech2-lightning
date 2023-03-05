class PreprocessDataset():
    def __init__(self) -> None:
        pass
    def __iter__(self):
        return self
    def __len__(self):
        raise NotImplementedError
    def __next__(self):
        raise NotImplementedError