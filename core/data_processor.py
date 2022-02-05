import math
import numpy as np
import pandas as pd
import MySQLdb                  # импортируем модуль для работы с БД MySql


class DataLoader():
    """A class for loading and transforming data for the lstm model"""

    def __init__(self, split, cols):
        self.dataframe = None
        self.conn = None
        self.cursor = None
        self.connection_to_db = False
        self.split = split
        self.cols = cols
        self.data_train = None
        self.data_test = None
        self.len_train = None
        self.len_test = None
        self.len_train_windows = None

    def split_data(self):
        i_split = int(len(self.dataframe) * self.split)
        self.data_train = self.dataframe.get(self.cols).values[:i_split]
        self.data_test  = self.dataframe.get(self.cols).values[i_split:]
        self.len_train  = len(self.data_train)
        self.len_test   = len(self.data_test)
        self.len_train_windows = None
        print("len(dataframe) == ", len(self.dataframe))
        print("len(self.data_train) == ", len(self.data_train))
        print("len(self.data_test) == ", len(self.data_test))

    def load_data_from_csv(self, filename):
        self.dataframe = pd.read_csv(filename, sep=",")
        self.dataframe.rename(
            columns={"<DATE>": "Date", "<TIME>": "Time", "<OPEN>": "Open", "<HIGH>": "High", "<LOW>": "Low",
                     "<CLOSE>": "Close", "<VOL>": "Volume"}, inplace=True)
        self.dataframe['Date'] = pd.to_datetime(self.dataframe['Date'], format='%Y%m%d')
        self.dataframe = self.dataframe.drop('Time', 1)
        self.split_data()

    def load_data_from_db(self, host, user, passwd, db, ticket, timeframe, how_many_bars):
        if not self.connection_to_db: self.connect_to_db(host, user, passwd, db)
        self.dataframe = self.get_share_data_from_db(ticket, timeframe, how_many_bars)
        self.split_data()

    def connect_to_db(self, host, user, passwd, db):
        try:
            self.conn = MySQLdb.connect(host=host, user=user, passwd=passwd, db=db)
            self.cursor = self.conn.cursor()
            self.connection_to_db = True
            print("Connection to db: OK")
        except MySQLdb.Error as ex:
            print("connection to DB failed, error code =", ex)
            quit()

    def get_share_data_from_db(self, ticket, timeframe, how_many_bars):
        table_name = ticket + "_" + timeframe
        self.cursor.execute(
            "SELECT time, open, high, low, close, volume FROM `" + table_name + "`" + " ORDER BY time DESC LIMIT " + str(how_many_bars)
        )

        # Get all data from table
        rows = self.cursor.fetchall()
        dataframe = pd.DataFrame(rows, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
        dataframe = dataframe[::-1].reset_index(drop=True)  # Reverse Ordering of DataFrame Rows + Reset index
        #print(dataframe.dtypes)
        return dataframe

    def de_normalise_predicted(self, price_1st, _data):
        return (_data + 1) * price_1st

    def get_last_data(self, seq_len, normalise):
        last_data = self.data_test[seq_len:]
        data_windows = np.array(last_data).astype(float)
        #data_windows = np.array([data_windows])
        #data_windows = self.normalise_windows(data_windows, single_window=False) if normalise else data_windows
        data_windows = self.normalise_windows(data_windows, single_window=True) if normalise else data_windows
        return data_windows

    def get_test_data(self, seq_len, normalise):
        '''
        Create x, y test data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise reduce size of the training split.
        '''
        data_windows = []
        for i in range(self.len_test - seq_len + 1):
            data_windows.append(self.data_test[i:i+seq_len])

        data_windows = np.array(data_windows).astype(float)
        data_windows = self.normalise_windows(data_windows, single_window=False) if normalise else data_windows

        x = data_windows[:, :-1]
        y = data_windows[:, -1, [0]]
        return x,y

    def get_train_data2(self, seq_len, normalise):
        '''
        Create x, y test data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise reduce size of the training split.
        '''
        data_windows = []
        for i in range(self.len_train - seq_len + 1):
            data_windows.append(self.data_train[i:i+seq_len])

        data_windows = np.array(data_windows).astype(float)
        data_windows = self.normalise_windows(data_windows, single_window=False) if normalise else data_windows

        x = data_windows[:, :-1]
        y = data_windows[:, -1, [0]]
        return x,y

    def get_train_data(self, seq_len, normalise):
        '''
        Create x, y train data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise use generate_training_window() method.
        '''
        data_x = []
        data_y = []
        for i in range(self.len_train - seq_len + 1):
            x, y = self._next_window(i, seq_len, normalise)
            data_x.append(x)
            data_y.append(y)
        return np.array(data_x), np.array(data_y)

    def generate_train_batch(self, seq_len, batch_size, normalise):
        '''Yield a generator of training data from filename on given list of cols split for train/test'''
        i = 0
        while i < (self.len_train - seq_len + 1):
            x_batch = []
            y_batch = []
            for b in range(batch_size):
                if i >= (self.len_train - seq_len + 1):
                    # stop-condition for a smaller final batch if data doesn't divide evenly
                    yield np.array(x_batch), np.array(y_batch)
                    i = 0
                x, y = self._next_window(i, seq_len, normalise)
                x_batch.append(x)
                y_batch.append(y)
                i += 1
            yield np.array(x_batch), np.array(y_batch)

    def _next_window(self, i, seq_len, normalise):
        '''Generates the next data window from the given index location i'''
        window = self.data_train[i:i+seq_len]
        window = self.normalise_windows(window, single_window=True)[0] if normalise else window
        x = window[:-1]
        y = window[-1, [0]]
        return x, y

    def normalise_windows(self, window_data, single_window=False):
        '''Normalise window with a base value of zero'''
        normalised_data = []
        window_data = [window_data] if single_window else window_data
        for window in window_data:
            normalised_window = []
            for col_i in range(window.shape[1]):
                normalised_col = [((float(p) / float(window[0, col_i])) - 1) for p in window[:, col_i]]
                normalised_window.append(normalised_col)
            normalised_window = np.array(normalised_window).T # reshape and transpose array back into original multidimensional format
            normalised_data.append(normalised_window)
        return np.array(normalised_data)