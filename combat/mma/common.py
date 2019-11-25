from scipy import stats
import time
import requests
from bs4 import BeautifulSoup
import re
import string

mma_data_location = r'E:\sports\mma'
max_tries = 5
starting_rating = 1000
rating_k_factor = 100
rating_floor = 100
rating_ceiling = 10000
rating_d = 1000
k_min_sensitivity = 1


def parse_list_of_ints_from_str(s):
    return [int(i) for i in re.findall('\d+', str(s))]


def clean_text(s):
    exclude = set(string.punctuation)
    return ''.join([i for i in str(s).lower() if i not in exclude])


def sleep_on_error():
    sleep_random_amount(min_time=1800, max_time=3600)


def sleep_normal():
    sleep_random_amount(min_time=1.0, max_time=3.0)


def sleep_random_amount(min_time=.05, max_time=.2, mu=None, sigma=1.0, verbose=False):
    if not mu:
        mu = (max_time + min_time)/2

    var = stats.truncnorm(
        (min_time - mu) / sigma, (max_time - mu) / sigma, loc=mu, scale=sigma)
    sleep_time = var.rvs()
    if verbose:
        print('Sleeping for {0} seconds: {0}'.format(sleep_time))
    time.sleep(sleep_time)


def pad_num(n, length):
    n_str = str(n)
    while len(n_str) < length:
        n_str = '0' + n_str
    return n_str


def get_session():
    session = requests.Session()
    session.headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.100 Safari/537.36'}
    return session


def get_soup(url, session = None):
    sleep_normal()

    if not session:
        session = get_session()

    r = session.get(url)
    soup = BeautifulSoup(r.text, 'lxml')
    return soup

def get_new_rating(rating1, rating2, outcome, multiplier = 1, rating_type = 0):
    '''
    :param rating1:
    :param rating2:
    :param outcome:
    :param multiplier:
    :return:
    '''

    if rating_type == 0:
        expected_outcome = rating1 / (rating1 + rating2)
        next_rating = rating1 + (multiplier * rating_k_factor * (outcome - expected_outcome))

    elif rating_type == 1:
        expected_outcome1 = 1 / (1 + 10 ** ((rating1 - rating2) / (rating1 + rating2)))
        next_rating = rating1 + (multiplier * rating_k_factor * (outcome-expected_outcome1))

    elif rating_type == 2:
        if outcome == 1:
            if rating1  < rating2:
                next_rating = rating2 + rating_k_factor
            else:
                next_rating= rating1 + rating_k_factor
        else:
            if rating2  < rating1:
                next_rating = rating2 - rating_k_factor
            else:
                next_rating= rating1 - rating_k_factor

    elif rating_type == 3:
        if outcome == 1:
            if rating1 <= (rating2 + k_min_sensitivity):
                next_rating = (rating1 + rating_k_factor + starting_rating)/2
            else:
                next_rating = (rating1 + starting_rating) / 2
        else:
            if (rating1 + k_min_sensitivity) >= rating2:
                next_rating = (rating1 - rating_k_factor + starting_rating)/2
            else:
                next_rating = (rating1 + starting_rating) / 2
    else:
        print('invalid rating_type: {rating_type}'.format(rating_type=rating_type))
        raise NotImplementedError()
    next_rating = max(next_rating, rating_floor)
    next_rating = min(next_rating, rating_ceiling)
    return next_rating

if __name__ == '__main__':
    print(get_new_rating(1000, 100, 1))
    print(get_new_rating(100, 1000, 1))

