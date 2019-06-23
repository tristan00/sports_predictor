import requests
import random
from bs4 import BeautifulSoup
import wikipedia
import time
from scipy import stats
import pandas as pd
import uuid
import urllib
from urllib.parse import urljoin
import re
from itertools import permutations

start_pages = {'https://en.wikipedia.org/wiki/Chael_Sonnen',
               'https://en.wikipedia.org/wiki/Holly_Holm',
               'https://en.wikipedia.org/wiki/Tyson_Fury',
               'https://en.wikipedia.org/wiki/Demetrious_Johnson_(fighter)'}

mma_page_section_names = ['mixed martial arts record']
boxing_page_section_names = ['professional boxing record']
professional_record_names = ['professional record']

data_location = r'C:\Users\trist\OneDrive\Desktop\mma_data'
base_url = 'https://en.wikipedia.org/'
country_url = 'https://en.wikipedia.org/wiki/List_of_countries_and_dependencies_by_population'
start_run_time = time.time()

pro_key = 'professional'
amateur_key = 'amateur'
exhibition_key = 'exhibition'
data_table_key = 'data'
other_key = 'other'
mma_key = 'mma'
boxing_key = 'boxing'
kickboxing_key = 'kickboxing'

pro_summary_table_text = ['professional record summary', 'professional record breakdown']
amateur_summary_table_text = ['amateur record summary', 'amateur record breakdown']
exhibition_summary_table_text = ['exhibition record summary', 'exhibition record breakdown']



def get_section_name_variations(initial_list):
    suffixes = ['[edit]', '(Incomplete)', ' ', ' ']
    prefixes = ['amateur', 'professional', ' ', ' ']
    input_list = suffixes + prefixes + initial_list
    output_list = [''.join(l) for i in range(len(suffixes)) for l in permutations(input_list, i + 1)]
    output_list = [i for i in output_list if [j for j in initial_list if j in i]]
    # print(output_list)
    return output_list

mma_page_section_names = get_section_name_variations(mma_page_section_names)
boxing_page_section_names = get_section_name_variations(boxing_page_section_names)
professional_record_names = get_section_name_variations(professional_record_names)


def get_country_urls():
    r = requests.get(country_url)
    soup = BeautifulSoup(r.text)
    a_tags = soup.find_all('a')
    links = []

    for i in a_tags:
        try:
            links.append(i['href'])
        except:
            pass

    # print(links)
    return links


links_to_avoid = get_country_urls()


def clean_text(s):
    return str(s).replace('|', ' ')


def sleep_random_amount(min_time=1.0, max_time=5.0, mu=None, sigma=1.0, verbose=False):
    if not mu:
        mu = (max_time + min_time)/2

    var = stats.truncnorm(
        (min_time - mu) / sigma, (max_time - mu) / sigma, loc=mu, scale=sigma)
    sleep_time = var.rvs()
    if verbose:
        print('Sleeping for {0} seconds: {0}'.format(sleep_time))
    time.sleep(sleep_time)



def get_row_num_of_headers(table):
    words_to_match = ['Res.', 'Record', 'Opponent', 'Type', 'Date', 'Location', 'Notes', 'Method']

    result = dict()
    for c, i in enumerate(table.find_all('tr')):
        row_text = i.get_text()
        result[c] = len([j for j in words_to_match if j in row_text])
    result_reversed = {j: i for i, j in result.items()}
    if result:
        return result_reversed[max(result.values())]
    return 0


def get_table_type(t):
    table_text = t.get_text().lower()

    if pro_key in table_text and [i for i in pro_summary_table_text if i in table_text] and t.name == 'table':
        return pro_key
    if amateur_key in table_text and [i for i in amateur_summary_table_text if i in table_text] and t.name == 'table':
        return amateur_key
    if exhibition_key in table_text and [i for i in exhibition_summary_table_text if i in table_text] and t.name == 'table':
        return exhibition_key
    if [i for i in ['record', 'opponent', 'method'] if i in t.get_text().lower()] and t.name == 'table':
        return data_table_key
    return other_key


def extract_table(f_url, tables_dict, general_stats):
    dfs = []
    new_urls = []

    fighter_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f_url))

    sport_keys = tables_dict.keys()
    for s in sport_keys:
        types_of_events = tables_dict[s].keys()
        for t in types_of_events:
            for j in tables_dict[s][t]:
                tr_tags = j.find_all('tr')
                header_index = get_row_num_of_headers(j)

                opponent_col_name = [i.get_text().strip() for i in tr_tags[header_index].find_all(['th', 'td']) if 'opponent' in i.get_text().strip().lower()][0]
                index_of_opponent = [c for c, i in enumerate(tr_tags[header_index].find_all(['th', 'td'])) if opponent_col_name.strip() == i.get_text().strip()][0]

                id_mapping = dict()
                for k in tr_tags[header_index + 1:]:

                    opponent_cell = k.find_all('td')[index_of_opponent]
                    opponent_a_tag = opponent_cell.find_all('a')
                    opponent_name = opponent_cell.get_text().strip()

                    opponent_rel_links = [k2['href'] for k2 in opponent_a_tag if k2['href'] not in links_to_avoid]

                    if opponent_rel_links:
                        opponent_rel_link = opponent_rel_links[-1]
                        opponent_abs_link = urljoin(base_url, opponent_rel_link)
                        new_urls.append(opponent_abs_link)
                        opponent_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, opponent_abs_link))
                        has_wiki = 1

                    else:
                        opponent_id = str(uuid.uuid4())
                        has_wiki = 0
                    id_mapping[opponent_name] = {'id': opponent_id, 'has_wiki': has_wiki}

                df = pd.read_html(str(j), header=header_index)[0]
                df['fighter_id'] = fighter_id
                df['opponent_id'] = df.apply(lambda x: id_mapping.get(x[opponent_col_name], {'id':str(uuid.uuid4())})['id'], axis = 1)
                df['opponent_has_wiki'] = df.apply(lambda x: id_mapping.get(x[opponent_col_name], {'has_wiki':0})['has_wiki'], axis = 1)

                df['sport'] = s
                df['event_type'] = t

                for g in general_stats:
                    df[g] = general_stats[g]

                df = df.applymap(lambda x: clean_text(x))
                dfs.append(df)
    if dfs:
        df = pd.concat(dfs)
    else:
        df = pd.DataFrame()

    return df, new_urls


def get_general_info(soup):
    output = dict()

    if soup:
        tr_tags = soup.find_all('tr')
        tr_tags = [i for i in tr_tags if i.find('th') and i.find('td')]
        for i in tr_tags:
            output[i.find('th').get_text()] = i.find('td').get_text()
    return output


def guess_sport(data, full_text):
    data_lc = clean_text(str(data)).lower()
    print(data_lc)
    kickboxing_refs1 = len(re.findall('kickbox', data_lc))
    boxing_refs1 = len(re.findall('[\W.,]+box', data_lc))
    mma_refs1 = len(re.findall('(mixed martial arts)|(mma)', data_lc))

    print(kickboxing_refs1, boxing_refs1, mma_refs1)

    if kickboxing_refs1 > boxing_refs1 and kickboxing_refs1 > mma_refs1:
        return kickboxing_key
    if mma_refs1 > boxing_refs1 and mma_refs1 > kickboxing_refs1:
        return mma_key
    if boxing_refs1 > mma_refs1 and boxing_refs1 > kickboxing_refs1:
        return boxing_key

    data_lc = clean_text(str(full_text)).lower()
    kickboxing_refs2 = len(re.findall('kickbox', data_lc))
    boxing_refs2 = len(re.findall('[\W.,]+box', data_lc))
    mma_refs2 = len(re.findall('(mixed martial arts)|(mma)', data_lc))
    print(kickboxing_refs2, boxing_refs2, mma_refs2)

    if kickboxing_refs2 > boxing_refs2 and kickboxing_refs2 > mma_refs2:
        return kickboxing_key
    if mma_refs2 > boxing_refs2 and mma_refs2 > kickboxing_refs2:
        return mma_key
    if boxing_refs2 > mma_refs2 and boxing_refs2 > kickboxing_refs2:
        return boxing_key

    return other_key


def scrape_fighter(next_url):
    sections_dict = dict()

    r = requests.get(next_url)
    soup = BeautifulSoup(r.text)
    stats_table_card = soup.find('table', {'class': 'infobox vcard'})
    general_stats = get_general_info(stats_table_card)

    mw_parser_output = soup.find('div', {'class': 'mw-parser-output'})
    if mw_parser_output:
        page_items = mw_parser_output.find_all(['h2', 'h3', 'table'])

        sections = dict()
        active_key = None
        for i in page_items:
            if i.name == 'h2':
                active_key = clean_text(i.get_text()).lower()
                sections[active_key] = []
            if active_key:
                sections[active_key].append(i)

        for i in sections:
            if 'mma' in i.lower() or 'boxing' in i.lower() or 'record' in i.lower():
                if i not in mma_page_section_names and i not in boxing_page_section_names and i not in ['Possibly missed key: Amateur kickboxing career[edit]', 'Possibly missed key: Professional boxing career[edit]', 'Possibly missed key: Mixed martial arts career[edit]', 'Possibly missed key: Kickboxing record (Incomplete)[edit]']:
                    print('Possibly missed key: {0}'.format(i))

        active_key2 = None
        for c1, i in enumerate(sections):

            if i in mma_page_section_names:
                sections_dict.setdefault(mma_key, dict())
                sections_dict[mma_key].setdefault(exhibition_key, list())
                sections_dict[mma_key].setdefault(pro_key, list())
                sections_dict[mma_key].setdefault(amateur_key, list())
                active_key1 = mma_key
            elif i in boxing_page_section_names:
                sections_dict.setdefault(boxing_key, dict())
                sections_dict[boxing_key].setdefault(exhibition_key, list())
                sections_dict[boxing_key].setdefault(pro_key, list())
                sections_dict[boxing_key].setdefault(amateur_key, list())
                active_key1 = boxing_key
            elif i in professional_record_names:
                sport_type = guess_sport(stats_table_card, r.text)
                print('guessing sport: {0} {1}'.format(next_url, sport_type))
                active_key1 = sport_type
                sections_dict.setdefault(sport_type, dict())
                sections_dict[sport_type].setdefault(amateur_key, list())
                sections_dict[sport_type].setdefault(pro_key, list())
                sections_dict[sport_type].setdefault(amateur_key, list())
            else:
                continue

            for c2, j in enumerate(sections[i]):
                data_type = get_table_type(j)
                # print(data_type)
                if data_type == other_key:
                    continue
                if data_type == data_table_key and c2 < 2:
                    active_key2 = pro_key
                if data_type in [pro_key, amateur_key, exhibition_key]:
                    active_key2 = data_type
                if data_type == data_table_key and active_key2:
                    sections_dict[active_key1][active_key2].append(j)
            active_key2 = None

    return sections_dict, general_stats


def scrape():
    #bfs is more reliable

    dfs = []

    pages_to_search = start_pages.copy()
    # pages_to_search = {'https://en.wikipedia.org/wiki/Lee_Swaby'}
    next_iteration_pages_to_search = set()
    searched_pages = set()

    iteration = 1

    while pages_to_search or next_iteration_pages_to_search:
        while pages_to_search:

            try:
                sleep_random_amount()
                next_url = pages_to_search.pop()
                print(next_url)

                searched_pages.add(next_url)

                sections_dict, general_stats = scrape_fighter(next_url)

                new_df, new_urls = extract_table(next_url, sections_dict, general_stats)
                if new_df.shape[0]:
                    dfs.append(new_df)
                    new_mma_df = new_df[new_df['sport'] == mma_key]
                    new_boxing_df = new_df[new_df['sport'] == boxing_key]
                    print('scaped url: {0},  mma records: {1},  boxing records: {2}'.format(next_url, new_mma_df.shape[0],
                                                                                            new_boxing_df.shape[0]))
                next_iteration_pages_to_search = next_iteration_pages_to_search | {i for i in new_urls if i not in searched_pages and i not in pages_to_search}

                if dfs:
                    df = pd.concat(dfs)
                    print('full df shape: {0}'.format(df.shape))
                    mma_df = df[df['sport'] == mma_key]
                    boxing_df = df[df['sport'] == boxing_key]
                    print('iteration: {0}, mma_records: {1}, boxing records: {2}, searched_pages: {3}, pages_to_search: {4}, next_iteration_pages_to_search: {5}'.format(iteration, mma_df.shape[0], boxing_df.shape[0], len(searched_pages), len(pages_to_search), len(next_iteration_pages_to_search)))
                    df.to_csv('{0}/raw_{1}.csv'.format(data_location, int(start_run_time)))
            except:
                import traceback
                traceback.print_exc()

        pages_to_search = next_iteration_pages_to_search.copy()
        next_iteration_pages_to_search = set()
        iteration += 1


if __name__ == '__main__':
    scrape()







