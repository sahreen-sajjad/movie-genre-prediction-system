import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit

GENRES = {"['drama']": 'drama', "['horror']": 'horror', "['thriller']": 'thriller', "['comedy']": 'comedy', "['documentary']": 'documentary', "['comedy', 'romance']": 'romcom'}

def load_data():
    """Reproduce the original sampled movie-ID joins."""
    from convokit import Corpus, download
    corpus = Corpus(filename=download('movie-corpus'))
    utterances = corpus.get_utterances_dataframe().sample(n=20000, random_state=42)
    conversations = corpus.get_conversations_dataframe().rename(columns={'meta.movie_idx':'meta.movie_id'})
    speakers = corpus.get_speakers_dataframe().rename(columns={'meta.movie_idx':'meta.movie_id'})
    conversations['meta.genre'] = conversations['meta.genre'].map(GENRES)
    conversations = conversations.dropna(subset=['meta.genre']).drop(columns=['meta.release_year','meta.rating','meta.votes'])
    speakers = speakers.drop(columns=['meta.credit_pos','meta.gender','meta.character_name'])
    return conversations.merge(utterances, on='meta.movie_id').merge(speakers, on='meta.movie_id').sample(n=10000, random_state=42)

def split_data(df, grouped=False):
    """Return train, test, validation; grouped mode holds out entire movies."""
    def split(frame, fraction):
        if grouped:
            splitter = GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=0)
            a,b = next(splitter.split(frame, groups=frame['meta.movie_id']))
        else:
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=fraction, random_state=0)
            a,b = next(splitter.split(frame, frame['meta.genre']))
        return frame.iloc[a].copy(), frame.iloc[b].copy()
    train, rest = split(df, .4)
    test, val = split(rest, .5)
    return train, test, val
