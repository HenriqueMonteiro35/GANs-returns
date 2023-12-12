### Generate and save images

from config import *

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from pandas.plotting import autocorrelation_plot
from pandas import Series

def plot_loss(G_loss, D_loss, choice, epoch, max_epoch):
    fig = plt.figure(figsize=(4,4))
    fig.suptitle(f'{choice.upper()} loss during training')
    plt.plot(D_loss, label='D loss')
    plt.plot(G_loss, label='G loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    if epoch + 1 == max_epoch:
        plt.savefig(os.path.join("models", choice, f'{choice}_{max_epoch}_epochs.png'))
    plt.show()



def plot_hist(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        # plt.plot(predictions[i, :, 0])
        # plt.axis('off')
        plt.hist(predictions[i, :, 0]) ## temp

    # plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()



from scipy import stats
def plot_uniform_test(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    def chisq_test(v):
        size = len(v)
        unit = int(np.sqrt(size))
        df = unit - 1
        cls = np.floor(v * unit)
        cls = cls[(cls>=0)&(cls<unit)]
        _, counts = np.unique(cls, return_counts=True)
        s = np.sum((counts - size/unit)**2/(size/unit))
        return 1-stats.chi2.cdf(s,df = df)

    predictions = model(test_input, training=False)
    ls1 = []; ls2 = []
    for i in range(predictions.shape[0]):
        ls1.append(stats.kstest(predictions[i, :, 0], 'uniform')[1])
        ls2.append(chisq_test(predictions[i, :, 0]))

    f, (ax1, ax2) = plt.subplots(2, 1, sharey=True, figsize=(4,4))
    ax1.set_title('p-values of KS test')
    ax1.hist(ls1)
    ax2.set_title('p-values of chisq test')
    ax2.hist(ls2)
    f.tight_layout()

    plt.show()



def plot_mean(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)
    mean = tf.reduce_mean(predictions,axis=[1,2])
    fig = plt.figure(figsize=(4,4))
    plt.scatter(mean,tf.zeros_like(mean))
    # plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()



def plot_var(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)
    mean = np.var(predictions,axis=(1,2))
    fig = plt.figure(figsize=(4,4))
    plt.scatter(mean,np.zeros_like(mean))
    # plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()



def autocorrelation_plt(dataframe):
    """
    Function to compute autocorrelation plot

    :param dataframe: dataframe (of floats) of time-series
    :output: plot with y axis as autocorrelation and x axis as lag
    """

    fig = plt.figure()
    fig.set_figwidth(10)
    fig.set_figheight(8)

    for i in dataframe:
        _ = autocorrelation_plot(dataframe[i], label=i)
    _ = plt.legend(loc='upper right')



def plot_histogram(predictions):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).

    fig = plt.figure(figsize=(16,16))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        # plt.plot(predictions[i, :, 0])
        # plt.axis('off')
        plt.hist(predictions[i, :, 0],bins= 25 ) ## temp

    # plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()



# Comparing the distribution of real and generated (synthetic) data
# PDFs show the probability of observing a value within a specific range
def empirical_pdf(real, syntethic):
    fig, (ax2) = plt.subplots(1, 1, figsize=(14,9))
    ax2.hist(real, density=True, bins=200,alpha=0.5,label='Real returns distribution')
    ax2.hist(syntethic, density=True, bins=200,alpha=0.5,label='Synthetic returns distribution')
    ax2.legend(loc='upper right')

    textstr = '\n'.join((
        r'%s' % ("- Real",),
        # t'abs_metric=%.2f' % abs_metric
        r'$skewness=%.2f$' % (skew(real),),
        r'$kurtosis=%.2f$' % (kurtosis(real),))
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax2.text(
        -8, 0.5, textstr,
        fontsize=14,
        verticalalignment='top',
        bbox=props
    )
    textstr = '\n'.join((
        r'%s' % ("- Synthetic",),
        # t'abs_metric=%.2f' % abs_metric
        r'$skewness=%.2f$' % (skew(syntethic),),
        r'$kurtosis=%.2f$' % (kurtosis(syntethic),))
    )
    ax2.text(
        -8, 0.3, textstr,
        fontsize=14,
        verticalalignment='top',
        bbox=props
    )

def get_highlow(log_r, intervals=None):
    log_r = np.array(log_r, dtype=np.float64)
    if intervals is None:
        intervals = np.array([2**x for x in range(10+1)]) #[5, 21, 63, 252]
    high = np.zeros((len(intervals), log_r.shape[0], log_r.shape[1]))
    low = np.zeros((len(intervals), log_r.shape[0], log_r.shape[1]))
    for i, interval in enumerate(intervals):
        for j, ret in enumerate(log_r.T):
            prices = np.exp(np.cumsum(ret/100))
            series = Series(prices).rolling(interval)
            high[i,:,j] = series.max()
            low[i,:,j] = series.min()
    return intervals, high, low

def parkinson(high, low):
    return np.sqrt(np.mean(
        1/(4 * np.log(2)) * np.power(np.log(high/low), 2), axis=0
    ))

def get_vols(log_r, intervals=None):
    log_r = np.array(log_r)
    if intervals is None:
        intervals = np.array([2**x for x in range(10+1)])
    vol = np.zeros((len(intervals), log_r.shape[1]))
    for i, interval in enumerate(intervals):
        for j, ret in enumerate(log_r.T):
            series = Series(ret/100).rolling(interval).sum()[interval-1:]
            vol[i,j] = series.std()
    return intervals, vol

def plot_parkinson(intervals, log_r, ticker_list, dtype="R", show=True):
    _, high, low = get_highlow(log_r, intervals=intervals)
    parks = np.array([parkinson(high[i][interval-1:], low[i][interval-1:]) \
                        for i, interval in enumerate(intervals)])
    _, vols = get_vols(log_r, intervals=intervals)
    for i, ticker in enumerate(ticker_list):
        plt.plot(intervals, parks.T[i]/vols[:,i], "*-",label=f"{dtype} {ticker}")

    plt.title("Parkinson constant for returns over N days")
    plt.ylabel("Parkinson constant")
    plt.xlabel("Interval of N days")
    plt.xscale("log")
    plt.legend()
    if show:
        plt.show()

def plot_vols(intervals, log_r, ticker_list, dtype="R", show=True):
    _, vol = get_vols(log_r, intervals=intervals)
    mult = np.sqrt(252/intervals)
    for i, ticker in enumerate(ticker_list):
        plt.plot(intervals, mult*vol[:,i], "*-", label=f"{dtype} {ticker}")

    plt.title("Annualized volatility for returns over N days")
    plt.ylabel("Volatility (%)")
    plt.xlabel("Interval of N days")
    plt.xscale("log")
    plt.legend()
    if show:
        plt.show()

# CMD distance: https://stats.stackexchange.com/questions/14673/measures-of-similarity-or-distance-between-two-covariance-matrices
def cmd(A, B):
    A, B = A.to_numpy(), B.to_numpy()
    return 1 - np.trace(np.matmul(A, B)) / (np.linalg.norm(A, ord="fro") * np.linalg.norm(B, ord="fro"))