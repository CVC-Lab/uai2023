#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  1 11:57:26 2018

@author: matteo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sts
import math

def bootstrap_ci(x, conf=0.95, resamples=10000):
    means = [np.mean(x[np.random.choice(x.shape[0], size=x.shape[0], replace=True), :], axis=0) for _ in range(resamples)]
    low = np.percentile(means, (1-conf)/2 * 100, axis=0)
    high = np.percentile(means, (1 - (1-conf)/2) * 100, axis=0)
    return low, high

def nonparam_ti(x, conf=0.95, prop=0.95):
    srt = np.sort(x, axis=0)
    n = x.shape[0]
    nu = n - sts.binom.ppf(conf, n, prop)
    if nu <= 1:
        raise ValueError('T.I. does not exist')
    if nu % 2 == 0:
        nu_1 = nu_2 = int(nu / 2)
    else:
        nu_1 = int(nu / 2  - 1 / 2)
        nu_2 = nu_1 + 1
    low = srt[nu_1 - 1, :]
    high = srt[n - nu_2, :]
    return low, high

def read_data(path, iters=None, default_batchsize=100, scale='Eps'):
    df = pd.read_csv(path, encoding='utf-8')
    if iters: df = df.loc[:iters, :]
    if not 'AvgRet' in df: df['AvgRet'] = df['EpRewMean']
    if not 'EpsThisIter' in df: df['EpsThisIter'] = df['BatchSize'] 
    df['EpsSoFar'] = np.cumsum(df['EpsThisIter'])
    if 'SamplesThisIter' in df: df['SamplesSoFar'] = np.cumsum(df['SamplesThisIter'])
    df['CumAvgRet'] = np.cumsum(df['AvgRet']*df[scale+'ThisIter'])/np.sum(df[scale+'ThisIter'])
    return df

def moments(dfs):
    concat_df = pd.concat(dfs, axis=1)
    mean_df = pd.concat(dfs, axis=1).groupby(by=concat_df.columns, axis=1).mean()
    std_df = pd.concat(dfs, axis=1).groupby(by=concat_df.columns, axis=1).std()
    return mean_df, std_df

def plot_all(dfs, key='AvgRet', ylim=None, scale='Samples'):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for df in dfs:
        value = df[key]
        ax.plot(df[scale+'SoFar'], value)
    return fig

def plot_ci(dfs, conf=0.95, key='AvgRet', ylim=None, scale='Eps', bootstrap=False, resamples=10000):
    n_runs = len(dfs)
    mean_df, std_df = moments(dfs)
    mean = mean_df[key]
    std = std_df[key]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(mean_df[scale+'SoFar'], mean)
    if bootstrap:
        x = np.array([df[key] for df in dfs])
        interval = bootstrap_ci(x, conf, resamples)
    else:
        interval = sts.t.interval(conf, n_runs-1,loc=mean,scale=std/np.sqrt(n_runs))
    ax.fill_between(mean_df[scale+'SoFar'], interval[0], interval[1], alpha=0.3)
    if ylim: ax.set_ylim(ylim)
    return fig

def compare(candidates, conf=0.95, key='AvgRet', ylim=None, xlim=None, scale='Episodes', bootstrap=False, resamples=10000, roll=1, separate=False, opacity=1, tolerance=False, prop=0.95):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    entries = []
    if type(roll) is int:
        roll = [roll]*len(candidates)
    for i, candidate_name in enumerate(candidates):
        entries.append(candidate_name)
        dfs = candidates[candidate_name]
        dfs = [dfs[j].rolling(roll[i]).mean() for j in range(len(dfs))]
        n_runs = len(dfs)
        mean_df, std_df = moments(dfs)
        mean = mean_df[key]
        std = std_df[key]
        if not separate:
            ax.plot(mean_df[scale+'SoFar'], mean)   
            if bootstrap:
                x = np.array([df[key] for df in dfs])
                interval = bootstrap_ci(x, conf, resamples)
            elif tolerance:
                x = np.array([df[key] for df in dfs])
                interval = nonparam_ti(x, conf, prop)
            else:
                interval = sts.t.interval(conf, n_runs-1,loc=mean,scale=std/np.sqrt(n_runs))
                print(candidate_name, end=': ')
                print_ci(dfs, conf)
            ax.fill_between(mean_df[scale+'SoFar'], interval[0], interval[1], alpha=0.3)
        else:
            for d in dfs:
                ax.plot(d[scale+'SoFar'], d[key], color=colors[i], alpha=opacity)
    ax.legend(entries)
    leg = ax.get_legend()
    if separate:
        for i in range(len(entries)):
            leg.legendHandles[i].set_color(colors[i])
            leg.legendHandles[i].set_alpha(1)
    if ylim: ax.set_ylim(None,ylim)
    if xlim: ax.set_xlim(0,xlim)
    return fig

def plot_data(path, key='VanillaAvgRet'):
    df = pd.read_csv(path)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    mean = df[key]
    ax.plot(df['EpsSoFar'], mean)
    return fig

def print_ci(dfs, conf=0.95, key='CumAvgRet'):
    n_runs = len(dfs)
    mean_df, std_df = moments(dfs)
    total_horizon = np.sum(mean_df['EpLenMean'])
    mean = mean_df[key][len(mean_df)-1]
    std = std_df[key][len(mean_df)-1]
    interval = sts.t.interval(conf, n_runs-1,loc=mean,scale=std/np.sqrt(n_runs))
    print('%f \u00B1 %f\t[%f, %f]\t total horizon: %d' % (mean, std, interval[0], interval[1], int(total_horizon)))

def save_ci(dfs, key, name='foo', conf=0.95, path='.', rows=501, xkey='EpisodesSoFar', bootstrap=False, resamples=10000, mult=1., header=True):
    n_runs = len(dfs)
    mean_df, std_df = moments(dfs)
    mean = mean_df[key].values * mult
    std = std_df[key].values * mult + 1e-24
    if bootstrap:
        data = np.array([df[key] * mult for df in dfs])
        interval = bootstrap_ci(data, conf, resamples)     
    else:
        interval = sts.t.interval(conf, n_runs-1,loc=mean,scale=std/math.sqrt(n_runs))
    low, high = interval
    if rows is not None:
        mean = mean[:rows]
        low = low[:rows]
        high = high[:rows]
    xx = range(1,len(mean)+1) if xkey is None else mean_df[xkey]
    plotdf = pd.DataFrame({"iteration": xx, "mean" : mean, "low" : low, "up": high})
    plotdf = plotdf.iloc[0:-1:1]
    plotdf.to_csv(name + '.csv', index=False, header=header)
    
def save_ti(dfs, key, name='foo', conf=0.95, prop=0.95, path='.', rows=501, xkey='EpisodesSoFar', mult=1., header=True):
    mean_df, std_df = moments(dfs)
    mean = mean_df[key].values * mult
    x = np.array([df[key] for df in dfs])
    interval = nonparam_ti(x, conf=conf, prop=prop)
    low, high = interval
    if rows is not None:
        mean = mean[:rows]
        low = low[:rows]
        high = high[:rows]
    xx = range(1,len(mean)+1) if xkey is None else mean_df[xkey]
    plotdf = pd.DataFrame({"iteration": xx, "mean" : mean, "low" : low, "up": high})
    plotdf = plotdf.iloc[0:-1:1]
    plotdf.to_csv(name + '.csv', index=False, header=header)