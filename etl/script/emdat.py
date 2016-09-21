import pandas as pd
import numpy as np
import os
from ddf_utils.str import to_concept_id
from ddf_utils.index import create_index_file

# configuration of file path
source = '../source/Data.csv'
out_dir = '../../'


def extract_entities_country(data):
    country = data[['iso', 'country_name']].copy()
    country.columns = ['country', 'name']
    country['country'] = country['country'].map(to_concept_id)

    return country.drop_duplicates()


def extract_entities_disaster(data):
    disas = data[['disaster type']].copy()
    disas['disaster'] = disas['disaster type'].map(to_concept_id)
    disas.columns = ['name', 'disaster']

    return disas.drop_duplicates()[['disaster', 'name']]


def extract_concepts(data):
    concs = data.columns[4:]  # all continuous concepts
    concs = [*['Year', 'Name', 'disaster', 'country'], *concs]  # add discrete ones

    df = pd.DataFrame([], columns=['concept', 'name', 'concept_type'])
    df['name'] = concs
    df['concept'] = df['name'].map(to_concept_id)
    df['concept_type'] = 'measure'

    df['concept_type'].iloc[0] = 'time'
    df['concept_type'].iloc[1] = 'string'
    df['concept_type'].iloc[2] = 'entity_domain'
    df['concept_type'].iloc[3] = 'entity_domain'

    return df


def extract_datapoints(data):
    dps = data.drop('country_name', axis=1).copy()
    dps.columns = list(map(to_concept_id, dps.columns))
    dps = dps.rename(columns={'disaster_type': 'disaster', 'iso': 'country'})

    dps['country'] = dps['country'].map(to_concept_id)
    dps['disaster'] = dps['disaster'].map(to_concept_id)

    dps = dps.sort_values(by=['country', 'disaster', 'year']).set_index(['country', 'disaster', 'year'])

    for i, col in dps.items():
        df = col.reset_index()
        df = df.dropna()
        # data are number of occurrence, all integers
        df[i] = df[i].astype(int)
        yield i, df


if __name__ == '__main__':
    print('reading source data...')
    data = pd.read_csv(source, encoding='iso-8859-1', skiprows=1)
    data.columns = list(map(str.strip, data.columns))

    print('creating concept file...')
    concepts = extract_concepts(data)
    path = os.path.join(out_dir, 'ddf--concepts.csv')
    concepts.to_csv(path, index=False)

    print('creating entities files...')
    country = extract_entities_country(data)
    path = os.path.join(out_dir, 'ddf--entities--country.csv')
    country.to_csv(path, index=False)

    disas = extract_entities_disaster(data)
    path = os.path.join(out_dir, 'ddf--entities--disaster.csv')
    disas.to_csv(path, index=False)

    print('creating datapoints files...')
    for i, df in extract_datapoints(data):
        path = os.path.join(out_dir, 'ddf--datapoints--{}--by--country--disaster--year.csv'.format(i))
        df.to_csv(path, index=False)

    print('creating index files...')
    create_index_file(out_dir)

    print('Done.')
