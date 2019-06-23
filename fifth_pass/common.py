from scipy import stats
import time


mma_data_location = r'C:\Users\trist\OneDrive\Desktop\mma_data'
boxing_data_location = r'C:\Users\trist\OneDrive\Desktop\boxing_data'




def clean_text(s):
    return str(s).replace('|', ' ')


def sleep_random_amount(min_time=0.1, max_time=1.0, mu=None, sigma=1.0, verbose=False):
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