import quandl
from sklearn.linear_model import LinearRegression
import matplotlib
import matplotlib.pyplot as plt

import config

matplotlib.use('TkAgg')

# Getting data
df = quandl.get("EURONEXT/BNP", authtoken=config.TOKEN)

# Adding features
for i in range(5):
    df[i] = df.Last

for i in range(1, 5):
    df[i][i:] = df.Last[:-i]

# Splitting training and test sets
training = df.iloc[5:int(df.shape[0]*0.7)][[0, 1, 2, 3, 4]]
test = df.iloc[int(df.shape[0]*0.7):][[0, 1, 2, 3, 4]]

# Training the model
reg = LinearRegression().fit(training[[1, 2, 3, 4]], training[[0]])

# Test the model
predicts = reg.predict(test[[1, 2, 3, 4]])

print("Score : ", reg.score(test[[1, 2, 3, 4]], test[[0]]))

plt.plot(predicts)
plt.plot(test[0].values)
plt.show()

# Backtesting over the test set (4xx days)

initial_capital = 1000
assets = 0

for index, row in test.iterrows():
    # If we predict that the futur price of the asset will go up tomorrow we buy
    # otherwise we sell
    if reg.predict([row[[1, 2, 3, 4]]])[0] > row[1]:
        buy_assets = int(initial_capital / row[1])
        assets += buy_assets
        initial_capital -= buy_assets * row[1]
    else:
        initial_capital += assets * row[1]
        assets = 0


print("Final Capital :", initial_capital)
print("Final number of assets :", assets)
print("Final value : ", initial_capital + assets*test.iloc[-1][0])
