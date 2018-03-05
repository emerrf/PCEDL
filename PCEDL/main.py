import pandas as pd
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from keras.models import Sequential, load_model
from keras.initializers import he_normal
from keras.layers import Dense, Dropout
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU


# # Reproducible Results
# # https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
# import tensorflow as tf
# import random as rn
# os.environ['PYTHONHASHSEED'] = '0'
# np.random.seed(0)
# rn.seed(0)
# session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
# from keras import backend as K
# tf.set_random_seed(0)
# sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
# K.set_session(sess)


# Logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
pd.set_option('display.width', 1000)

WT_TYPES = {"WT1": "shore", "WT2": "shore", "WT3": "shore", "WT4": "shore",
            "WT5": "offshore", "WT6": "offshore"}

J53_VARNAMES = {"Sequence No.": "seqNo", "air density": "rho", "Humidity": "H",
                "S_a": "Sa", "S_b": "Sb", "y (% relative to rated power)": "y"}


def read_J53_data(dir):
    data_list = list()
    for dirpath, _, filenames in os.walk(dir):
        for f in filenames:
            fullpath = os.path.abspath(os.path.join(dirpath, f))
            logger.info("Reading data from: {}".format(fullpath))
            df = pd.read_csv(fullpath, header=0, sep=" ", index_col="Sequence No.")
            df.rename(columns=J53_VARNAMES, inplace=True)
            id = f.split("_")[1].split(".")[0]
            df["WT"] = id
            df["Type"] = WT_TYPES[id]

            data_list.append(df)
    data = pd.concat(data_list)
    return data


def run():

    dataset = "windpw"  # windpw or WT1

    if(dataset == "WT1"):
        # Load J53 Data
        J53_data = read_J53_data("data/J53")

        # Select WT1 data and get order to match R analysis
        wt_data = J53_data.loc[J53_data["WT"] == "WT1",
                               ["V", "D", "rho", "I", "Sb", "y"]]
        wt1_order = pd.read_csv("../R/WT1_seqNo_order.csv", header=0)
        wt_data = wt_data.loc[wt1_order["seqNo"], :]

    else:
        # Default dataset: windpw from kernplus R package
        wt_data = pd.read_csv("../R/kernplus_windpw.csv")

    ## Data Visualisation
    # import seaborn as sns
    # sns.set()
    # # g = sns.lmplot(x="V", y="y", hue="WT", data=J53_data)
    # g = sns.pairplot(wt_data)

    # Preprocess (no missing values)
    X = wt_data[["V", "D", "rho", "I", "Sb"]]
    Y = wt_data["y"]

    train_size = int(wt_data.shape[0] * 0.9)  # 42787
    test_size = wt_data.shape[0] - train_size  # 4755

    X_train = X.iloc[np.arange(train_size)]
    Y_train = Y.iloc[np.arange(train_size)]
    X_test = X.iloc[train_size + np.arange(test_size)]
    Y_test = Y.iloc[train_size + np.arange(test_size)]

    # Standardize
    X_train = (X_train - X_train.mean()) / X_train.std()
    X_test = (X_test - X_test.mean()) / X_test.std()

    # Model
    # He et al., http://arxiv.org/abs/1502.01852 (initializer)
    model = Sequential()
    model.add(Dense(64, input_dim=5,
                    kernel_initializer=he_normal(seed=1),
                    # kernel_regularizer=regularizers.l2(0.1),
                    activation='relu'))
    # model.add(Dropout(0.05))
    # model.add(Dense(32,
    #                 kernel_initializer=he_normal(seed=1),
    #                 # kernel_regularizer=regularizers.l2(0.1),
    #                 activation='relu'))
    # model.add(Dropout(0.05))
    model.add(Dense(1,
                    kernel_initializer=he_normal(seed=1)))
    opt = Adam(lr=0.00025, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0., amsgrad=False)

    model.compile(loss='mean_squared_error', optimizer=opt)
    print(model.summary())

    checkpoint = ModelCheckpoint("model.hdf5", monitor='val_loss', verbose=0,
                                 save_best_only=True, mode='min')

    # check 5 epochs
    early_stop = EarlyStopping(monitor='val_loss', patience=1000, mode='min',
                               min_delta=10, verbose=1)

    callbacks_list = [checkpoint]#, #early_stop]

    # history = model.fit(x, y, validation_data=(x_test, y_test), epochs=100,
    #                     callbacks=callbacks_list)


    history = model.fit(X_train, Y_train, epochs=3000, batch_size=3, verbose=1,
                        # validation_split=0.2,
                        validation_data=(X_test, Y_test),
                        callbacks=callbacks_list)

    # """
    # Best epoch: 996
    # Min Val MSE: 79.02872171010881
    # Min Val RMSE: 8.88980999291373
    # Test MSE: 79.02871965753147
    # Test RMSE: 8.889809877468217
    # Time: 160.28177285194397 # CPU
    # Time: 534.9832010269165  # GPU
    # Time: 444.0515308380127 # 24core
    #
    # """
    # """hist_64_batch30_nadam_henormal.png
    # Epoch: 2416
    # Min Val MSE: 120.34393005371093
    # Min Val RMSE: 10.970138105498533
    # Test MSE: 120.34392826142766
    # Test RMSE: 10.970138023809348
    # Time: 3623.863386631012
    # """

    # from keras.utils import plot_model
    # # Win Hack
    # graphviz_path = os.path.abspath(
    #     "C:/Users/emerrf/Anaconda3/envs/PCEDL/Library/bin/graphviz/;")
    # os.environ["PATH"] += graphviz_path
    # plot_model(model, show_shapes=True)
    #val_mse, val_mae = model.evaluate(X_test, Y_test, verbose=1)

    plot_model_history(history)

    best_model = load_model("model.hdf5")

    ypred = best_model.predict(X_test)
    y = Y_test.values.reshape(ypred.shape)
    mse = np.mean(np.power(y - ypred, 2))
    print("Test MSE: {} \nTest RMSE: {}".format(mse, np.sqrt(mse)))

def plot_model_history(history):
    hist = pd.DataFrame(history.history, index=history.epoch)
    hist["val_AMK"] = 133.0267; hist["AMK"] = 72.9650  # windpw
    #hist["val_AMK"] = 56.0239  #; hist["AMK"] = 72.9650
    hist_ax = hist.plot()
    hist_ax.set_ylim(50, 300) # windpw
    #hist_ax.set_ylim(0, 200)  # WT1
    best_epoch = hist[hist["val_loss"] == hist["val_loss"].min()]
    min_mse = best_epoch["val_loss"]
    print("Best epoch: {}\nMin Val MSE: {}\nMin Val RMSE: {}".format(
        best_epoch.index.values[0], min_mse.values[0], np.sqrt(min_mse.values[0])))
    hist_ax.get_figure().savefig("hist.png")



if __name__ == "__main__":
    import time
    start = time.time()
    run()
    print("Time: {}".format(time.time() - start))