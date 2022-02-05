#pip install numpy pandas tensorflow-gpu keras matplotlib mysqlclient

import os
import json
import time
import math
import matplotlib.pyplot as plt
from core.data_processor import DataLoader
from core.model import Model


def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()


def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    # Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()


def main():
    configs = json.load(open('config.json', 'r'))
    if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])

    # import pandas as pd
    # dataframe = pd.read_csv(os.path.join('data', configs['data']['filename']), sep=",")
    # dataframe.rename(
    #     columns={"<DATE>": "Date", "<TIME>": "Time", "<OPEN>": "Open", "<HIGH>": "High", "<LOW>": "Low",
    #              "<CLOSE>": "Close", "<VOL>": "Volume"}, inplace=True)
    # dataframe['Date'] = pd.to_datetime(dataframe['Date'], format='%Y%m%d')
    # dataframe = dataframe.drop('Time', 1)
    # print(dataframe)
    # exit(1)

    data = DataLoader(
        split=configs['data']['train_test_split'],
        cols=configs['data']['columns']
    )

    # to load data from CSV files
    # data.load_data_from_csv(
    #     filename=os.path.join('data', configs['data']['filename']),
    #
    # )

    ticket = "SBER"
    timeframe = "D1"

    # to load data from MySQL DB
    data.load_data_from_db(
        host=configs['data']['db_host'],
        user=configs['data']['db_login'],
        passwd=configs['data']['db_pass'],
        db=configs['data']['db_name'],
        ticket=ticket,  # "SBER",
        timeframe=timeframe,  # "D1",
        how_many_bars=5700
    )

    model = Model()
    model.build_model(configs)
    x, y = data.get_train_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise'] #False
    )

    x2, y2 = data.get_train_data2(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise']  #False
    )

    print("train data shapes: ", x.shape, y.shape)
    print("train data shapes: ", x2.shape, y2.shape)
    #print(x, y);print(x2, y2); exit(1)


    '''
    # in-memory training'''
    model.train(
        x,
        y,
        epochs = configs['training']['epochs'],
        batch_size = configs['training']['batch_size'],
        save_dir = configs['model']['save_dir'],
        timeframe=timeframe
    )
    '''

    # out-of memory generative training
    steps_per_epoch = math.ceil(
        (data.len_train - configs['data']['sequence_length']) / configs['training']['batch_size'])
    model.train_generator(
        data_gen=data.generate_train_batch(
            seq_len=configs['data']['sequence_length'],
            batch_size=configs['training']['batch_size'],
            normalise=configs['data']['normalise']
        ),
        epochs=configs['training']['epochs'],
        batch_size=configs['training']['batch_size'],
        steps_per_epoch=steps_per_epoch,
        save_dir=configs['model']['save_dir'],
        timeframe=timeframe
    )'''

    x_test, y_test = data.get_test_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise'] #False
    )

    print("test data shapes: ", x_test.shape, y_test.shape)
    #print(x_test, y_test); exit(1)
    #print(x_test, y_test, x_test.shape, y_test.shape)

    model.eval_test(x_test,  y_test, verbose=2)

    _ev = model.eval_test2(x_test, y_test, verbose=0)
    print("### ", _ev, " ###")

    last_data_2_predict = data.get_last_data(-(configs['data']['sequence_length']-1), configs['data']['normalise'])
    print("*** ", -(configs['data']['sequence_length']-1), last_data_2_predict.size, "***")

    predictions2 = model.predict_point_by_point(last_data_2_predict)
    #print(predictions2)

    last_data_2_predict_prices = data.get_last_data(-(configs['data']['sequence_length']-1), False)
    last_data_2_predict_prices_1st_price = last_data_2_predict_prices[0][0]
    predicted_price = data.de_normalise_predicted(last_data_2_predict_prices_1st_price, predictions2[0])
    print("!!!!!", predictions2, predicted_price, "!!!!!")

    # predictions = model.predict_sequences_multiple(x_test, configs['data']['sequence_length'], configs['data']['sequence_length'])
    # predictions = model.predict_sequence_full(x_test, configs['data']['sequence_length'])
    predictions = model.predict_point_by_point(x_test)

    # plot_results_multiple(predictions, y_test, configs['data']['sequence_length'])
    plot_results(predictions, y_test)


if __name__ == '__main__':
    main()
