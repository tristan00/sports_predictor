from scipy import stats
import time
import requests
from bs4 import BeautifulSoup
mma_data_location = r'C:\Users\trist\OneDrive\Desktop\mma_data'
boxing_data_location = r'C:\Users\trist\OneDrive\Desktop\boxing_data'




def clean_text(s):
    return str(s).replace('|', ' ')


def sleep_random_amount(min_time=2.0, max_time=10.0, mu=None, sigma=1.0, verbose=False):
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


def get_soup(url, session = None, sleep = True):
    if sleep:
        sleep_random_amount()

    if not session:
        session = get_session()

    r = session.get(url)
    soup = BeautifulSoup(r.text)
    return soup