import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent

class DataFrames:
    _ceo_comp = None
    
    @property
    def ceo_comp(self):
        if self._ceo_comp is None:
            self._ceo_comp = pd.read_excel(DATA_DIR / 'ceo_comp.xlsx')
        return self._ceo_comp

_data = DataFrames()

def ceo_comp():
    return _data.ceo_comp
