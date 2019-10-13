import requests
from bs4 import BeautifulSoup
from  urllib import parse
from common import (get_soup, get_session)
import scrapy


base_url = 'https://www.tapology.com/'
output_folder = r'E:\sports\mma\tapology_jsons'


def get_initial_urls():
    initial_url_lists =  ['https://www.tapology.com/rankings/current-top-ten-lightweight-mma-fighters-155-pounds',
                          'https://www.tapology.com/rankings/current-top-ten-featherweight-mma-fighters-145-pounds',
                          'https://www.tapology.com/rankings/current-top-ten-bantamweight-mma-fighters-135-pounds',
                          'https://www.tapology.com/rankings/35-top-flyweight-mma-fighters',
                          'https://www.tapology.com/rankings/1261-top-women-bantamweight-fighters',
                          'https://www.tapology.com/rankings/1262-top-women-flyweight-fighters',
                          'https://www.tapology.com/rankings/1263-top-women-strawweight-fighters',
                          'https://www.tapology.com/rankings/1264-top-women-atomweight-fighters',
                          'https://www.tapology.com/rankings/1265-top-women-featherweight-fighters',
                          'https://www.tapology.com/rankings/current-top-ten-best-pound-for-pound-mma-and-ufc-fighters',
                          'https://www.tapology.com/rankings/top-ten-fan-favorite-mma-and-ufc-fighters',
                          'https://www.tapology.com/rankings/33-current-best-pound-for-pound-female-mma-fighters',
                          'https://www.tapology.com/rankings/current-top-ten-heavyweight-mma-fighters-265-pounds',
                          'https://www.tapology.com/rankings/current-top-ten-light-heavyweight-mma-fighters-205-pounds',
                          'https://www.tapology.com/rankings/current-top-ten-middleweight-mma-fighters-185-pounds',
                          'https://www.tapology.com/rankings/current-top-ten-welterweight-mma-fighters-170-pounds'
                          ]

    output_urls = set()
    for url_ranking_page_url in initial_url_lists:
        soup = get_soup(url_ranking_page_url)
        name_soups = soup.find_all('div', {'class':'rankingItemsItemRow name'})
        next_batch_of_urls = set([parse.urljoin(base_url, i.find('a')) for i in name_soups if i.find('a')])
        output_urls.update(next_batch_of_urls)
    return output_urls



def scrape_url(url):
    output = []

    soup = get_soup(url)
    fighter_info_soup = soup.find('div', {'class':'details details_two_columns'})
    fighter_info_soup = fighter_info_soup.find('ul', recursive = False)
    fighter_info_soups = fighter_info_soup.find_all('li', recursive = False)
    fighter_info_dict = dict()
    for i in fighter_info_soups:
        label = i.find('strong').getText()
        value = i.find('span').getText()
        fighter_info_dict[label] = value


def run_scrape():
    url_batch = get_initial_urls()
    scraped_urls = set()

    while True:
        if not url_batch:
            break

        for url in url_batch:
            pass


def test():
    s = get_session()
    url = 'https://www.tapology.com/fightcenter/fighters/jon-jones-bones'
    response = s.get(url)
    soup = BeautifulSoup(response.text)

    fighter_info_soup = soup.find('div', {'class':'details details_two_columns'})
    fighter_info_soup = fighter_info_soup.find('ul', recursive = False)
    fighter_info_soups = fighter_info_soup.find_all('li', recursive = False)
    fighter_info_dict = dict()
    for i in fighter_info_soups:
        if i.find('strong') and i.find('span'):
            label = i.find('strong').getText()
            value = i.find('span').getText()
            fighter_info_dict[label] = value

    print(fighter_info_dict)

    data = {'Access-Control-Request-Headers':'authorization,content-type',
            'Access-Control-Request-Method':'GET',
            'DNT':'1',
            'Origin':'https://www.tapology.com',
            }
    r2 = s.options('https://api.tapology.com/v1/internal_fighters/8320275')
    r3 = s.get('https://api.tapology.com/v1/internal_fighters/8320275')
    a =1





if __name__ == '__main__':
    test()

    # run_scrape()

