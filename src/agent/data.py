from typing import Dict

import pandas as pd
import torch

FEATURE_PATHS = {
    'open': '~/python_project/data/open.pk',
    'close': '~/python_project/data/close.pk',
    'high': '~/python_project/data/high.pk',
    'low': '~/python_project/data/low.pk',
    'volume': '~/python_project/data/volume.pk',
    'vwap': '~/python_project/data/vwap.pk',
    'return': '~/python_project/data/return.pk',
}

class DataContainer:
    def __init__(self, device, feature_files: Dict[str,str]=FEATURE_PATHS, period=['2015-01-01', '2021-12-31']):
        self._data : Dict[str, torch.Tensor] = {}
        self._dates = None
        self._tickers = None

        for name, path in feature_files.items():
            df = pd.read_pickle(path)
            if period is not None:
                df = df.loc[period[0]:period[1]]
            if self._dates is None:
                self._dates = df.index
            if self._tickers is None:
                self._tickers = df.columns
            self._data[name.lower()] = torch.tensor(df.values, dtype=torch.float32, device=device)

        ones_tensor = torch.ones_like(next(iter(self._data.values())))
        self._data['1.0'] = ones_tensor

    def __getitem__(self, key: str):
        key = key.lower()
        if key not in self._data:
            raise KeyError(f"Feature '{key}' not found in data container.")
        return self._data[key]
    
    def keys(self):
        return self._data.keys()
        
    @property
    def dates(self):
        return self._dates
    
    @property
    def tickers(self):
        return self._tickers