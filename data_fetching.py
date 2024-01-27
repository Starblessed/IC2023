import pandas as pd
import numpy as np
import datetime

sljp_codes = ['CM0320', 'JC0341', 'JC0342', 'MR0361', 'MR0363', 'MR0369', 'TJ0303', 'TJ0306']
sljp_prefix = '01RJ20'


def combine_sheets(excel: pd.DataFrame, ano_i, ano_f) -> pd.DataFrame:
    in_df = pd.DataFrame()
    for ano in range(ano_i, ano_f):
        data = pd.read_excel(excel, str(ano))
        in_df = pd.concat([in_df, data])
    return in_df


def extract_by_code(frame: pd.DataFrame, prefix: str, codes: list) -> pd.DataFrame:
    frame = frame.copy()
    frame = frame[frame['Local'].str.contains(prefix)]
    selector = []
    for i, row in frame.iterrows():
        if row['Local'][6:] not in codes:
            selector.append(0)
        else:
            selector.append(1)
    frame['selector'] = selector
    return frame[frame['selector'] == 1]


def feature_extraction(dataframe: pd.DataFrame, params=[]) -> pd.DataFrame:
    frame = dataframe.copy()
    frame['Dia'] = frame['Data'].map(lambda x: x.day).astype('float32')
    frame['Mês'] = frame['Data'].map(lambda x: x.month).astype('float32')
    frame['Ano'] = frame['Data'].map(lambda x: x.year).astype('float32')
    frame['Sem'] = frame['Data'].map(lambda x: x.isocalendar()[1]).astype('float32')
    frame['DiaSem'] = frame['Data'].map(lambda x: x.isocalendar()[2]).astype('float32')
    if params != ['Tudo']:
        cols = ['Local', 'Dia', 'Mês', 'Ano', 'Sem', 'DiaSem']
        cols.extend(params)
        to_return = frame[cols]
    else:
        to_return = frame
    return to_return


def fetch_data(filename: str, params=[], i_year=2012, f_year=2023,
               prefix=sljp_prefix, codes=sljp_codes) -> pd.DataFrame:
    data = pd.ExcelFile(filename)
    combined = combine_sheets(data, i_year, f_year)
    raw_data = extract_by_code(combined, prefix, codes)
    extracted = feature_extraction(raw_data, params)
    return extracted


def write_columns(dataframe: pd.DataFrame, filename='col_names.txt') -> None:
    cols = sorted(set(dataframe.columns))
    col_names = open(filename, 'a', encoding='utf8')
    for i, col in enumerate(cols):
        col_names.write(f'{i}ª variável: {col} \n')
    col_names.close()


def join_duplicates(dataframe: pd.DataFrame, var: str, join_list: list) -> pd.DataFrame:
    frame = dataframe.copy()
    frame.fillna(0, inplace=True)
    dummy_name = '_dummy'
    frame[dummy_name] = 0

    for old_var in join_list:
        try:
            frame.replace(' ', 0, inplace=True)
        except:
            pass
        frame[old_var] = frame[old_var].astype('float32')
        frame[dummy_name] = frame[dummy_name] + frame[old_var]
        frame.drop(old_var, axis=1, inplace=True)
    frame.rename(columns={dummy_name:var}, inplace=True)

    return frame


def remove_outliers(base_df: pd.DataFrame, var: str):
    new_df = base_df.copy()
    new_df.sort_values(var, axis=0, inplace=True)
    rol = new_df[var].values

    quartile_1 = np.percentile(rol, 25)
    quartile_3 = np.percentile(rol, 75)

    IQR = quartile_3 - quartile_1
    w_min = quartile_1 - 1.5 * IQR
    w_max = quartile_3 + 1.5 * IQR

    to_drop = [0 if (w_min <= value <= w_max) else 1 for value in rol] 
    new_df['drop'] = to_drop
    drop_ones = new_df[(new_df['drop'] == 1)].index
    new_df.drop(drop_ones, axis=0, inplace=True)
    new_df.drop(['drop'], axis=1, inplace=True)
    return new_df


def date_sort(base_df: pd.DataFrame) -> pd.DataFrame:
    new_df = base_df.copy()
    dates = new_df['Data'].values
    new_dates = []
    for date in dates:
        new_dates.append(date.astype(datetime.datetime))
    new_df['Data'] = new_dates
    new_df.sort_values(by='Data', inplace=True)
    return new_df



def pluvio_query(filename: str, i_year=2012, f_year=2023) -> pd.DataFrame:
    full_df = pd.read_csv(filename)
    str_dates = full_df['Data'].values.tolist()
    
    dt_dates = [datetime.datetime.strptime(date, "%m/%d/%Y") for date in str_dates] # Converts str to datetime.datetime

    # Generates a list with 0s and 1s to determine which lines to be kept from the original DataFrame
    mask = pd.Series([True if ((date >= datetime.datetime(i_year, 1, 1)) & (date <= datetime.datetime(f_year, 12, 31)))
             else False for date in dt_dates])
    
    query_df = full_df.loc[mask.values]
    dt_dates = query_df['Data'].values.tolist()
    str_dates = [datetime.datetime.strptime(date, "%m/%d/%Y").strftime("%Y-%m-%d") for date in dt_dates]
    query_df['Data'] = str_dates

    return query_df

#
# AJUSTAR PARA UTILIZAR A MÉDIA DOS DIAS, MEDIANTE ANÁLISE EXPLORATÓRIA PRÉVIA
#
def merge_pluvio(base_df: pd.DataFrame, col: str, filename: str, i_year=2012, f_year=2023) -> pd.DataFrame:
    
    pluvio_df = pluvio_query(filename, i_year=i_year, f_year=f_year)
    new_df = base_df.copy()
    new_df['Data'] = new_df['Data'].astype('str')

    pluvio_df = pluvio_df[['Data', col]]
    pluvio_df = pluvio_df.groupby('Data').max().reset_index()
    pluvio_df['Data'] = pluvio_df['Data'].astype('str')
    
    new_df = new_df.merge(pluvio_df, how='left', on='Data')
    new_df.rename(columns={'24h': 'Prec. max em 24h (mm)'}, inplace=True)


    return new_df

