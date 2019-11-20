import quandl
from sklearn.linear_model import LinearRegression
import matplotlib
import matplotlib.pyplot as plt

import config

matplotlib.use('TkAgg')

# Some global variables

CAC40 = ['AC', 'AL', 'AIR', 'MT', 'ATO', 'CS',
         'BNP', 'EN',
         'CAP', 'CA', 'ACA',
         'BN',
         'ENGI', 'EL',
         'RMS',
         'KER',
         'LR', 'OR', 'MC'
         'ML',
         'ORA',
         'RI',
         'UG',
         'PUB',
         'RNO',
         'SAF', 'SGO', 'SAN', 'SU', 'GLE', 'SW',
         'DSY',
         'STM',
         'FTI', 'HO', 'FP'
         'URW',
         'VIE', 'DG', 'VIV']


def test_lin_reg_trading(asset="BNP", window=3, init_capital=1000, plot=True, debug_msg=True):
    # Getting data
    df = quandl.get("EURONEXT/" + asset, authtoken=config.TOKEN)

    # Adding features
    for i in range(window):
        df[i] = df.Last

    for i in range(1, window):
        df[i][i:] = df.Last[:-i]

    # Splitting training and test sets
    training = df.iloc[window:int(df.shape[0] * 0.7)][list(range(window))]
    test = df.iloc[int(df.shape[0]*0.7):][list(range(window))]

    # Training the model
    reg = LinearRegression().fit(training[list(range(1, window))], training[[0]])

    # Test the model
    predicts = reg.predict(test[list(range(1, window))])

    print("Score : ", reg.score(test[list(range(1, window))], test[[0]]))

    if plot:
        plt.plot(predicts)
        plt.plot(test[0].values)
        plt.title("Asset : " + asset)
        plt.show()

    # Backtesting over the test set (4xx days)

    current_capital = init_capital
    assets = 0

    for index, row in test.iterrows():
        # If we predict that the futur price of the asset will go up tomorrow we buy
        # otherwise we sell
        if reg.predict([row[list(range(1, window))]])[0] > row[1]:
            buy_assets = int(current_capital / row[1])
            assets += buy_assets
            current_capital -= buy_assets * row[1]
        else:
            current_capital += assets * row[1]
            assets = 0

    if debug_msg:
        print("Final Capital :", current_capital)
        print("Final number of assets :", assets)
        print("Final value : ", current_capital + assets * test.iloc[-1][0])
        print("Final return : ", (current_capital + assets * test.iloc[-1][0]) / init_capital)

    return (current_capital + assets * test.iloc[-1][0]) / init_capital


for asset in CAC40:
    test_lin_reg_trading(asset)