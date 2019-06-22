import requests
import random
from bs4 import BeautifulSoup
import wikipedia
import time
from scipy import stats


male_start_page = 'https://en.wikipedia.org/wiki/Travis_Fulton'
female_start_page = 'https://en.wikipedia.org/wiki/Valentina_Shevchenko_(fighter)'


def sleep_random_amount(min_time=1.0, max_time=4.0, mu=None, sigma=1.0, verbose=False):
    if not mu:
        mu = max_time - min_time

    var = stats.truncnorm(
        (min_time - mu) / sigma, (max_time - mu) / sigma, loc=mu, scale=sigma)
    sleep_time = var.rvs()
    if verbose:
        print('Sleeping for {0} seconds: {0}'.format(sleep_time))
    time.sleep(sleep_time)


def scrape():
    s = requests.Session()

    pages_to_search = [male_start_page, female_start_page]
    searched_pages = []

    while pages_to_search:
        next_url = random.choice(pages_to_search)
        searched_pages.append(searched_pages)

        r = s.get(next_url)
        soup = BeautifulSoup(r.text)
        soup.find()



