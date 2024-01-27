import re
import pandas as pd
import os
import numpy as np
import datetime as dt

leg_pattern = re.compile('\d{2}\/\d{2}\/\d{4}[ ]{2}\d{2}:\d{2}:\d{2}[ ][A-Za-z ]+[ ]\d+\.\d{1}[ ]{1,4}\d+\.\d{1}[ ]{1,4}\d+\.\d{1}[ ]{1,4}\d+\.\d{1}[ ]{1,4}\d+\.\d{1}/gm')

regexes = [
        '\d{2}\/\d{2}\/\d{4}',
        '\d{2}:\d{2}:\d{2}',
        '\d+\.\d{1}|ND'
        ]

sug_patterns = [re.compile(regex) for regex in regexes]
ND_PATTERN = re.compile('\d{2}\/\d{2}\/\d{4}[ ]{2}\d{2}:\d{2}:\d{2}[ ]+ND')


def txt_to_csv(txt_path: str):
    files = os.listdir(txt_path)
    try:
        for filename in files:
            df = pd.DataFrame(columns= ['Data', 'Hora', '15min', '01h', '04h', '24h', '96h'])
            text = open(os.path.join(txt_path, filename), 'r').read()

            dates = re.findall(sug_patterns[0], text)
            hours = re.findall(sug_patterns[1], text)
            values = re.findall(sug_patterns[2], text)
            values = [val if val != 'ND' else np.nan for val in values]
        
            dates_mmddaaaa = [dt.datetime.strptime(date, '%d/%m/%Y').strftime('%m/%d/%Y') for date in dates]

            df['Data'] = dates_mmddaaaa
            df['Hora'] = hours
            for iter, column in zip([0, 1, 2, 3, 4], df.columns[2:]):
                ete = [iter + i*5 for i in range(int(len(values)/5))]
                extracted = [values[i] for i in ete]
                df[column] = extracted
            df.to_csv(txt_path[:-3] + rf'csv\{filename[:-3]}csv', index=False)
    except Exception as e:
        print(f'Exception found while parsing file {filename}:')
        ND = re.findall(ND_PATTERN, text)
        if ND:
            print(f'Data missing at {ND[0][:21]}')
        else:
            print(repr(e))

    
def join_data(csv_path: str, station: str) -> None:
    files = os.listdir(csv_path)

    concatenated_df = pd.DataFrame()
    for file in files:
        file_path = os.path.join(csv_path, file)
        df = pd.read_csv(file_path)
        concatenated_df = pd.concat([concatenated_df, df], ignore_index=True)
    concatenated_df.to_csv(csv_path + r'\comprised_' + station + '.csv', index=False)
    