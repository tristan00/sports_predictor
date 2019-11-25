import requests
from bs4 import BeautifulSoup
from urllib import parse
from common import (get_soup, get_session)
import copy
import datetime
import os
import pandas as pd
import tqdm
import traceback
import time
import random
import pickle

base_url = 'https://www.sherdog.com/'
base_output_folder = r'/media/td/Samsung_T5/sports/mma/sherdog_data'

initial_url = {'https://www.sherdog.com/fighter/Henry-Cejudo-125297',
               'https://www.sherdog.com/fighter/Max-Holloway-38671',
               'https://www.sherdog.com/fighter/Khabib-Nurmagomedov-56035',
               'https://www.sherdog.com/fighter/Kamaru-Usman-120691',
               'https://www.sherdog.com/fighter/Israel-Adesanya-56374',
               'https://www.sherdog.com/fighter/Jon-Jones-27944',
               'https://www.sherdog.com/fighter/Stipe-Miocic-39537',
               'https://www.sherdog.com/fighter/Weili-Zhang-186663',
               'https://www.sherdog.com/fighter/Valentina-Shevchenko-45384',
               'https://www.sherdog.com/fighter/Amanda-Nunes-31496'
               }


def scrape_url(url, iteration):
    fighter_urls = set()
    soup = get_soup(url)

    personal_dict = dict()
    bio_soup = soup.find('div', {'class': 'bio'})

    name_meta_soup = soup.find('meta', {'itemprop': 'name'})
    description_meta_soup = soup.find('meta', {'itemprop': 'name'})

    if name_meta_soup and name_meta_soup.has_attr('content'):
        personal_dict['sherdog_name'] = name_meta_soup['content']
    if not personal_dict['sherdog_name'] and name_meta_soup:
        personal_dict['sherdog_name'] = name_meta_soup.getText()

    if description_meta_soup and description_meta_soup.has_attr('content'):
        personal_dict['sherdog_description'] = description_meta_soup['content']
    if not personal_dict['sherdog_description'] and description_meta_soup:
        personal_dict['sherdog_description'] = description_meta_soup.getText()

    personal_dict['fighter_id'] = url
    personal_dict['scrape_iteration'] = iteration

    if bio_soup:
        birth_date_soup = bio_soup.find('span', {'itemprop': 'birthDate'})
        if birth_date_soup:
            personal_dict['birth_date'] = birth_date_soup.getText()

        nationality_soup = bio_soup.find('strong', {'itemprop': 'nationality'})
        if nationality_soup:
            personal_dict['nationality'] = nationality_soup.getText()

        height_soup = bio_soup.find('strong', {'itemprop': 'height'})
        if height_soup:
            personal_dict['height'] = height_soup.getText()

        weight_soup = bio_soup.find('strong', {'itemprop': 'weight'})
        if weight_soup:
            personal_dict['weight'] = weight_soup.getText()

    fights = list()

    sections = soup.find_all('section')
    fight_counter = 0
    for section in sections:

        fight_type_text = ''
        first_div = section.find('div', {'class': 'module fight_history'})
        if first_div:
            fight_type_soup = section.find('div', {'class': 'module_header'})
            if fight_type_soup:
                fight_type_text = fight_type_soup.getText()
                if '-' in fight_type_text:
                    fight_type_text = fight_type_text.split('-')[1]

            table_div_soup = section.find('div', {'class': 'content table'})
            table_soup = table_div_soup.find('table')
            tr_tags = table_soup.find_all('tr')[1:]

            for tr_tag in tr_tags:
                td_tags = tr_tag.find_all('td')
                if len(td_tags) == 6:
                    result = td_tags[0].getText()
                    if result in ['Result', 'Date']:
                        continue

                    opponent_name = td_tags[1].getText()
                    opponent_id = copy.copy(opponent_name)
                    opponent_has_url = False
                    if td_tags[1].find('a'):
                        opponent_id = parse.urljoin(base_url, td_tags[1].find('a')['href'])
                        fighter_urls.add(opponent_id)
                        opponent_has_url = True

                    date_str = ''
                    if td_tags[2].find('span', {'class': 'sub_line'}):
                        date_str = td_tags[2].find('span', {'class': 'sub_line'}).getText()

                    event_url = None
                    event_name = None
                    event_has_url = False
                    if td_tags[2].find('a'):
                        event_name = td_tags[2].find('a').getText()
                        event_url = parse.urljoin(base_url, td_tags[2].find('a')['href'])
                        event_has_url = True

                    referee = td_tags[3].find('span', {'class': 'sub_line'}).getText()
                    method = td_tags[3].getText().replace(referee, ' ')
                    fight_end_round = td_tags[4].getText()
                    fight_end_time = td_tags[5].getText()

                    fight_dict = {'result': result,
                                  'fighter_id': url,
                                  'opponent_name': opponent_name,
                                  'opponent_id': opponent_id,
                                  'opponent_has_url': opponent_has_url,
                                  'fight_date': date_str,
                                  'event_name': event_name,
                                  'event_url': event_url,
                                  'event_has_url': event_has_url,
                                  'referee': referee,
                                  'method': method,
                                  'fight_end_round': fight_end_round,
                                  'fight_end_time': fight_end_time,
                                  'fight_type_text': fight_type_text,
                                  'fight_counter': fight_counter
                                  }
                    fight_counter += 1

                    fights.append(fight_dict)
    return {'fight_data': fights,
            'personal_data': [personal_dict],
            'fighter_urls': fighter_urls}


def save_data(personal_data, fight_data, output_folder, urls_to_scrape, scraped_urls, iteration, counter, batch, run_id):
    t1 = time.time()
    print(
        f'''Run id: {run_id}. Timestamp: {datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}, iteration {iteration}. Num of fights scraped: {len(fight_data)}. Num of fighters scraped: {len(personal_data)}. Num of fighter urls found: {len(urls_to_scrape - scraped_urls)}. Scraped {counter} of {len(batch)} urls in batch.''')
    print('saving data, {}'.format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))

    personal_df = pd.DataFrame.from_dict(personal_data)
    fight_df = pd.DataFrame.from_dict(fight_data)

    personal_df.to_csv(f'{output_folder}/personal_data.csv', sep='|', index=False)
    fight_df.to_csv(f'{output_folder}/fight_data.csv', sep='|', index=False)

    with open(f'{output_folder}/urls_to_scrape.pkl', 'wb') as f:
        pickle.dump(urls_to_scrape, f)
    with open(f'{output_folder}/scraped_urls.pkl', 'wb') as f:
        pickle.dump(scraped_urls, f)
    with open(f'{output_folder}/iteration.pkl', 'wb') as f:
        pickle.dump(iteration, f)
    print('saved data in {} seconds'.format(time.time() - t1))
    print()


def run_scrape(run_id=None, max_iterations = 20):
    if not run_id:
        now = datetime.datetime.now()
        run_id = now.strftime('%Y-%m-%d_%H-%M-%S')
        print(f'Starting new scrape: {run_id}')
        output_folder = f'{base_output_folder}/{run_id}'

        if os.path.exists(output_folder):
            os.removedirs(output_folder)

        os.mkdir(output_folder)

        urls_to_scrape = copy.copy(initial_url)
        scraped_urls = set()
        personal_data = list()
        fight_data = list()
        iteration = 0

    else:
        print(f'Resuming saved scrape: {run_id}')
        output_folder = f'{base_output_folder}/{run_id}'
        personal_df = pd.read_csv(f'{output_folder}/personal_data.csv', sep='|')
        fight_df = pd.read_csv(f'{output_folder}/fight_data.csv', sep='|')

        personal_data = personal_df.to_dict(orient='records')
        fight_data = fight_df.to_dict(orient='records')

        with open(f'{output_folder}/urls_to_scrape.pkl', 'rb') as f:
            urls_to_scrape = pickle.load(f)
        with open(f'{output_folder}/scraped_urls.pkl', 'rb') as f:
            scraped_urls = pickle.load(f)
        with open(f'{output_folder}/iteration.pkl', 'rb') as f:
            iteration = pickle.load(f)

    while iteration < max_iterations:
        urls_to_scrape = urls_to_scrape - scraped_urls
        url_batch = set(list(urls_to_scrape))
        print()
        print(f'starting iteration: {iteration}, batch size: {len(url_batch)}')
        print()

        if not url_batch:
            break

        for url in tqdm.tqdm(url_batch):
            scraped_urls.add(url)
            try:
                res_dict = scrape_url(url, iteration)
                personal_data.extend(res_dict['personal_data'])
                fight_data.extend(res_dict['fight_data'])
                urls_to_scrape.update(res_dict['fighter_urls'])
            except:
                traceback.print_exc()
                time.sleep(300)
        print()
        save_data(personal_data, fight_data, output_folder, urls_to_scrape, scraped_urls, iteration, None, url_batch, run_id)
        iteration += 1
        print()

    return run_id


if __name__ == '__main__':
    # run_id='2019-10-12_16-19-56'
    run_id = '2019-10-13_14-32-23'
    run_scrape(run_id=run_id)
