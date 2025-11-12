import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt


dates = []
prices = []

def get_data(filename): #will fill the empty lists with data
    with open(filename, 'r') as csvfile:
        csvfilereader = csv.reader(csvfile) #reader method to read csv file
        next(csvfilereader) #to skip the first row
        for i, row in enumerate(csvfilereader):
            dates.append(i)  # simple numeric index (0, 1, 2, ...), will add the index to dates list
            prices.append(float(row[4]))  # use 'Close' column, not 'Open', append the price to the empty list
    return

def predict_prices(dates, prices, x):
    dates = np.reshape(dates, (len(dates), 1)) #to reshape into a n-by-1 format

    svr_lin = SVR(kernel='linear', C=1e3) #kernel=type of svm
    svr_poly = SVR(kernel='poly', C=1e3, degree = 2)
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)

    svr_rbf.fit(dates, prices)
    svr_lin.fit(dates, prices)
    svr_poly.fit(dates,prices)

    plt.scatter(dates, prices, color='black', label='Data')
    plt.plot(dates, svr_rbf.predict(dates), color='red', label='RBF Model')
    plt.plot(dates, svr_lin.predict(dates), color='green', label='Linear Model')
    plt.plot(dates, svr_poly.predict(dates), color='blue', label='Poly Model')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show(block=True)

    return svr_rbf.predict(x)[0], svr_lin.predict(x)[0], svr_poly.predict(x)[0]

get_data('AAPL.csv')
print('Loaded', len(dates), 'rows')
predicted_prices = predict_prices(dates, prices, 5)
print(predicted_prices)


