""" tests for functions in main.py """
import nltk.corpus
import pandas as pd
import pytest

import main
import numpy as np

# OBTAINED EXPECTED


""" tests for get_reviews """
@pytest.mark.get_reviews
class TestGetReviews:
    def test_n_200(self):
        """
        test retrieving 200 reviews
        """
        appid = 1245620  # Elden Ring (has 100,000+ reviews)
        assert len(main.get_reviews(appid, 200)) == 200

    def test_n_67(self):
        """
        test retrieving 67 reviews
        """
        appid = 1245620  # Elden Ring (has 100,000+ reviews)
        obtained = main.get_reviews(appid, 67)
        print(obtained)
        assert len(obtained) == 67

    def test_n_135(self):
        """
        test retrieving 135 reviews
        """
        appid = 1245620  # Elden Ring (has 100,000+ reviews)
        assert len(main.get_reviews(appid, 135)) == 135

    def test_n_0(self):
        """
        test retrieving 0 reviews
        """
        appid = 1245620  # Elden Ring (has 100,000+ reviews)
        assert len(main.get_reviews(appid, 0)) == 0

    def test_n_negative(self):
        """
        test for when num_reviews is negative
        """
        appid = 1245620  # Elden Ring (has 100,000+ reviews)
        assert len(main.get_reviews(appid, -50)) == 0

    def test_n_greater_than_reviews(self):
        """
        Test for when we request a greater number of reviews than there are available
        Not the best test condition, but the main thing is just to make sure it doesn't break
        """
        appid = 899930  # Spaceball
        assert len(main.get_reviews(appid, 2**100)) > 10


""" tests for reviews_to_df """
@pytest.mark.reviews_to_df
class TestReviewsToDf:
    test_reviews = [{'recommendationid': '116441553', 'author': {'steamid': '76561198030328270', 'num_games_owned': 723, 'num_reviews': 21, 'playtime_forever': 95, 'playtime_last_two_weeks': 95, 'playtime_at_review': 95, 'last_played': 1654290296}, 'language': 'english', 'review': "Hatsune Miku on PC? Then what're you waiting for...", 'timestamp_created': 1654290282, 'timestamp_updated': 1654290282, 'voted_up': True, 'votes_up': 0, 'votes_funny': 0, 'weighted_vote_score': 0, 'comment_count': 0, 'steam_purchase': True, 'received_for_free': False, 'written_during_early_access': False}, {'recommendationid': '116441131', 'author': {'steamid': '76561198249441296', 'num_games_owned': 86, 'num_reviews': 23, 'playtime_forever': 2187, 'playtime_last_two_weeks': 2187, 'playtime_at_review': 2187, 'last_played': 1654289379}, 'language': 'english', 'review': 'I;m tinking miku miku ooo eee ooo', 'timestamp_created': 1654289624, 'timestamp_updated': 1654289624, 'voted_up': True, 'votes_up': 0, 'votes_funny': 0, 'weighted_vote_score': 0, 'comment_count': 0, 'steam_purchase': True, 'received_for_free': False, 'written_during_early_access': False}, {'recommendationid': '116440040', 'author': {'steamid': '76561198024803504', 'num_games_owned': 67, 'num_reviews': 3, 'playtime_forever': 567, 'playtime_last_two_weeks': 567, 'playtime_at_review': 551, 'last_played': 1654288773}, 'language': 'english', 'review': "Been playing project diva for a long long time (from PSP) I love to be able to finally play this game in PC, somehow feels more comfortable.\nSomething I would love to see is the option to uncap FPS and that world's end dance hall received the same senbonzakura treatment (there are two different versions of the song added to the game)\nAnyway, for my review I'd say it is a great game with great nostalgic songs, the jump between hard and extreme is kind of tough but still it is really enjoyable.", 'timestamp_created': 1654287899, 'timestamp_updated': 1654287899, 'voted_up': True, 'votes_up': 0, 'votes_funny': 0, 'weighted_vote_score': 0, 'comment_count': 0, 'steam_purchase': True, 'received_for_free': False, 'written_during_early_access': False}]

    def test_is_dataframe(self):
        assert type(main.reviews_to_df(self.test_reviews)) == pd.DataFrame

    def test_column_names(self):
        obtained = main.reviews_to_df(self.test_reviews).columns.values.tolist()
        expected = ['recommendationid', 'language', 'review', 'timestamp_created',
                    'timestamp_updated', 'voted_up', 'votes_up', 'votes_funny',
                    'weighted_vote_score', 'comment_count', 'steam_purchase',
                    'received_for_free', 'written_during_early_access', 'steamid',
                    'num_games_owned', 'num_reviews', 'playtime_forever',
                    'playtime_last_two_weeks', 'playtime_at_review', 'last_played']
        assert obtained == expected

    def test_unnested_data(self):
        """
        In the returned json there is a nested dictionary accessible by the key 'author'. This test is to see if the
        data inside has been added to the dataframe as individual columns
        """
        obtained = main.reviews_to_df(self.test_reviews).loc[2, 'recommendationid']
        expected = '116440040'
        assert obtained == expected

    def test_nested_data(self):
        obtained = main.reviews_to_df(self.test_reviews).loc[0, 'steamid']
        expected = '76561198030328270'
        assert obtained == expected


""" tests for raw_frequency_count """
@pytest.mark.raw_frequency_count
class TestRawFrequencyCount:
    test_reviews = [
        "This game sucks!!!",
        "ONE OF THE GREATEST GAMES EVER MADE, I LOVE THIS GAME",
        "I've played this game for 700 hours and haven't gotten past the first area. This is one of the biggest games ever",
        "awesome game but can someone help me im stuck"
    ]
    test_series = pd.Series(test_reviews)
    stop_words = set(nltk.corpus.stopwords.words('english'))

    obtained = main.raw_frequency_count(test_series, stop_words)

    def test_type(self):
        assert type(self.obtained) == nltk.FreqDist

    def test_lowered(self):
        # test all words have been converted to lower case
        assert all(x == x.lower() for x in list(self.obtained))

    def test_isalpha(self):
        # test all numbers and punctuation have been removed
        assert all(x.isalpha() for x in list(self.obtained))

    def test_example(self):
        obtained = dict(self.obtained)
        assert obtained['game'] == 4


""" tests for lem_frequency_count """
@pytest.mark.lem_frequency_count
class TestLemFrequencyCount:
    test_reviews = [
        "This game sucks!!!",
        "ONE OF THE GREATEST GAMES EVER MADE, I LOVE THIS GAME",
        "I've played this game for 700 hours and haven't gotten past the first area. This is one of the biggest games ever",
        "awesome game but can someone help me im stuck"
    ]
    test_series = pd.Series(test_reviews)
    stop_words = set(nltk.corpus.stopwords.words('english'))

    obtained_general = main.lem_frequency_count(test_series, stop_words)

    def test_type(self):
        assert type(self.obtained_general) == nltk.FreqDist

    def test_lowered(self):
        # test all words have been converted to lower case
        assert all(x == x.lower() for x in list(self.obtained_general))

    def test_isalpha(self):
        # test all numbers and punctuation have been removed
        assert all(x.isalpha() for x in list(self.obtained_general))

    def test_example(self):
        # tests that 'games' is lemmatized to 'game'
        obtained = dict(self.obtained_general)
        assert obtained['game'] == 6

    def test_do_not_lem(self):
        do_not_lem = ['games']
        obtained = dict(main.lem_frequency_count(self.test_series, self.stop_words, do_not_lem))
        assert obtained['games'] == 2











