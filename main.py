import quandl
import config

df = quandl.get("EURONEXT/BNP", authtoken=config.TOKEN)

training = df.loc[:'2018-11-18']
test = df.loc['2018-11-18':]

for i in range(5):
    training[i] = training.Last

for i in range(1, 5):
    training[i][i:] = training.Last[:-i]

print(training[['Last', 0, 1, 2, 3, 4]])