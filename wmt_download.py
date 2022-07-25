"""WMT14: Translate dataset."""

import datasets
from torch.utils.data import DataLoader

#_LANGUAGE_PAIRS = [(lang, "en") for lang in ["de", "fr"]]

wmt16_dataset_de_en = datasets.load_dataset('wmt16', 'de-en')

train_dataloader = DataLoader(wmt16_dataset_de_en['train'], batch_size=8, shuffle=True)
test_dataloader = DataLoader(wmt16_dataset_de_en['test'], batch_size=8, shuffle=True)

train_labels = next(iter(train_dataloader))
print(f"Labels batch shape: {train_labels}")