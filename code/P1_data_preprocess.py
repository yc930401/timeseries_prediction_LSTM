import pandas as pd
from matplotlib import pyplot

dataset = pd.read_csv('Beijing_PM25_raw.csv', delimiter=',', index_col='datetime',
                      parse_dates={'datetime': [1, 2, 3, 4]},
                      date_parser=lambda x: pd.datetime.strptime(x, '%Y %m %d %H'))

dataset.drop('No', axis=1, inplace=True)
dataset.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
# mark all NA values with 0
dataset['pollution'].fillna(0, inplace=True)
# drop the first 24 hours
dataset = dataset[24:]

values = dataset.values
# specify columns to plot
groups = [0, 1, 2, 3, 5, 6, 7]
i = 1
# plot each column
pyplot.figure()
for group in groups:
    pyplot.subplot(len(groups), 1, i)
    pyplot.plot(values[:, group])
    pyplot.title(dataset.columns[group], y=0.5, loc='right')
    i += 1
pyplot.show()

dataset.to_csv('Beijing_PM25_processed.csv')
