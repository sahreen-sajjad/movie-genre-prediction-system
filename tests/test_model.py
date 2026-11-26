import unittest
import pandas as pd
from src.data import split_data
from src.model import build_model

class TestPipeline(unittest.TestCase):
    def test_grouped_split(self):
        df = pd.DataFrame({'meta.movie_id': list(range(20))*2, 'meta.genre':['drama']*40})
        splits = split_data(df, grouped=True)
        groups = [set(s['meta.movie_id']) for s in splits]
        self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    def test_unseen_text(self):
        model = build_model().fit([['hello'],['goodbye']], ['drama','comedy'])
        self.assertIn(model.predict([['unknown']])[0], ['drama','comedy'])
