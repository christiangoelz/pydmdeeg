#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 11:02:30 2020

@author: christiangolz
"""


import numpy as np
import pandas as pd
import sys
import copy as cp
import warnings 
from math import floor
from numpy import matrix, diag
from numpy.linalg import cond, eig, pinv, norm
from scipy.linalg import svd
import matplotlib.pyplot as plt
import seaborn as sns
from optht import optht

class DMD: 
    """EEG signal decomposition using Dynamic Mode Decomposition (DMD).

    This object can be used to decompose EEG data to spatio-temporal coherent 
    patterns with DMD. Basically DMD is an algorithm that allows to approximate 
    the relation of all signals in pairs of consecutive time instances
    arranged in matrix X of size n x m, where n denotes sensors and m denotes measurement points.
    Base
    
    Parameters
    ----------
    X : array, shape (n_epochs, n_channels, n_times) | shape (n_channels, n_times)
        the data 
    y : array, shape (n_epochs)
        Target vector relative to X.
    channels : list, size: n_channels 
        list with channel names of same size as X[2].
    dt : int
        1/samplingrate.
    stacking_factor : int, optional
        stacking factor h for delay delay embedding technique see [1] for details.
        If not provided no stacking will be applied. The Default is 0.  
    win_size : int, optional
        windowsize over which DMD is computed. The default is 100.
    overlap : int, optional
        overlap of sliding window . The default is 0.
    datascale : str, {'none', 'centre', 'centre_norm', 'norm'}
        defines if X should be scaled before
        none - no sclaing 
        centre - mean centering 
        norm - normalize 
        centre_norm - z transform 
        The default is 'norm'.
    algorithm : str, {'exact', 'standard'}
        algorithm to use see [2] for details. The default is 'exact'.
    truncation : dict, {'method': {'cut', 'optht'}, {keep: n}, optional
        use of SVD truncation in first step.  If 'method' is 'optht' hard threshiolding based on 
        [3] and implemented with [4] is used. If 'method':'cut' and 'keep':n n SVD modes are kept. 
        The default is None.

    Attributes
    ----------
    AmpCh_Err: dict
        Contains Amplitude and Error as defined in [1] 
    X: array, shape (n_epochs, n_channels, n_times)
        the data - scaled if scalingw as applied
    Stats: dict, optional 
        Stats calculated for defined Freqeuncy band based on mode_stats()
    dt: float 
        dt of snapshots defined by 1/samplingrate  
    info: dict 
        contains information of parameter set/used in DMD
    results: dict
        results of DMD containing raw Psi, Amplitudes, Lambda, Mu assiciated 
        with info about window, label and trial 
    sclaed: bool 
        info about datascale 
    y: array, shape(ntrials,)
        triallabels
        
    Methods
    ----------
    DMD_win(self): 
        Runs DMD in defined windows 
    getAmp(self, fband, labels = []): 
        methdod to extract Amplitude of Modes in Dataframe in defined frequencies
        filtered after label
    getPSI(self, fband, labels = []): 
        methdod to extract Magnitude of Modes in Dataframe in defined frequencies
        filtered after label
    mode_stats(self,fband, labels = []): 
        calculaes descriptive stats over all modes if defined frequencies filtered 
        after label. Takes mode magnitudes (default) or mode amplitudes as basis
        
    $$$ PLOTING METHODS: $$$
    plot_statsCh(self):
        plot Channel statistics based on DMD.mode_stats() 
    plot_frRPsi(self, labels = []):
        plot DMD spectrim for each label 
    plot_frRLam(self, labels = []):
        plot lambda spectrum for each label
    plot_ChAmpErr(self):
        plot Channel Amplitude and Error of Approximation based on DMD modes and 
        lambda 
        
    References
    ----------
    .. [1]  Brunton, B. W., Johnson, L. A., Ojemann, J. G., & Kutz, J. N. (2016). 
            Extracting spatial–temporal coherent patterns in large-scale neural 
            recordings using dynamic mode decomposition. Journal of Neuroscience 
            Methods,258, 1–15. http://doi.org/10.1016/J.JNEUMETH.2015.10.010
            
    .. [2]  Tu, J. H. (2014). On dynamic mode decomposition: Theory and applications. 
            Journal of Computational Dynamics. http://doi.org/10.3934/jcd.2014.1.391
            
    .. [3]  Donoho, D., & Gavish, M. (2013). The Optimal Hard Threshold for 
            Singular Values is 4/sqrt(3). IEEE Transactions on Information Theory, 60. 
            http://doi.org/10.1109/TIT.2014.2323359
            
    .. [4]  Erichson, B (2019). Optimal Hard Threshold for Matrix Denoising - 
            last access: 15.05.2020 under https://github.com/erichson/optht 
    """
    def __init__(self, X = None, y = None, channels = None, dt = None, stacking_factor = 0, win_size = 100, 
                 overlap = 0, datascale = 'norm', algorithm = 'exact', truncation = {'method': None, 'keep': None}):

        
        #Check inputs
        _check_option('datascale', datascale, ['none', 'centre', 'centre_norm', 'norm'], extra='')
        _check_option('algorithm', algorithm, ['exact', 'standard'], extra='')
        
        
        if X is None: 
            warnings.warn('No data provided, specifiy data matrix')
        elif y is None and len(X.shape) == 3:
            raise ValueError('No trial labels provided')
        elif channels is None: 
            raise ValueError('No channel information provided')
        
        #get data structure
        n_chan = len(channels)
        chan_num = X.shape[1] if len(X.shape) ==3 else X.shape[0]
        if n_chan != chan_num:
            raise ValueError("Number channel doesn't match")
        trials = X.shape[0] if len(X.shape) ==3 else 1
        len_trial = int(X.size/n_chan/trials)
        numws =  int(floor(len_trial-overlap)/(win_size-overlap))
                       
        #Define object vars
        self.X = X
        self.y = y
        self.dt = dt
        self.scaled = False 
        self.info = {'trials': trials, 
                     'dt' : dt,
                     'trials_pts' : len_trial,
                     'channels' : channels,
                     'n_chan' : n_chan,
                     'numws_per_trial' : numws, 
                     'stacking_factor': stacking_factor,
                     'win_size' : win_size,
                     'overlap' : overlap,
                     'datascale' :datascale,
                     'algorithm' : algorithm, 
                     'truncation' : truncation}
        

    def DMD_win(self):
        """
        calculates DMD in windows defined in class with overlap defined in class
        
        Returns
        -------
        Attribute
        results -  
        AmpCh_Err: dict
            Contains Amplitude and Error as defined in [1] 
        results: dict
            results of DMD containing raw Psi, Amplitudes, Lambda, Mu assiciated 
            with info about window, label and trial 
        """
        if self.scaled != True:
            self._scale_input()            
        X = self.X
        y =self.y
        ws = self.info['win_size']
        ol = self.info['overlap']
        trials = self.info['trials']
        numws = self.info['numws_per_trial']
        n_chan = self.info['n_chan']
        channels = self.info['channels']
                 
        Window = []
        Trial =  []
        Label =  []
        Lambda = []
        Mu = []
        Psi = []
        Amp = []
        AppErr = []
        FroErW = []
        AmpCh = []
        err_lab = []
        
        sys.stdout.write('Decomposing data in windows of {} samples with {} samples overlap:\n'.format(ws,ol))
        for triali in range(trials):
            if trials > 1:
                Xep = X[triali,:,:]
            else: 
                Xep = X
            
            #print out processing status 
            t = (triali + 1)/trials
            sys.stdout.write('\r')
            sys.stdout.write("[%-20s] %d%%" % ('='*int(20*t), 100*t))
            sys.stdout.flush()
            
            for win in range(numws):
                start = win*(ws-ol)
                end = (win+1)*ws-win*ol;
                XYwin = Xep[:, start:end]
                
                psi, lam, Xhat, Xaug, condXaug, z0, amp = _DMD_comp(XYwin, self.info['stacking_factor'],
                                                                   self.info['algorithm'], self.info['truncation'] )
                
      
                
                apperr = np.amax(abs(XYwin[:,:(ws-ol)]-Xhat.real[:n_chan,:(ws-ol)]), axis = 1)
                froerw = norm(XYwin[:,:(ws-ol)]-Xhat.real[:n_chan,:(ws-ol)], 'fro')/norm(XYwin[:,:(ws-ol)],'fro')
                ampch = np.amax(XYwin[:,:(ws-ol)], axis = 1) - np.amin(XYwin[:,:(ws-ol)], axis = 1)
                AppErr.append(apperr)                
                FroErW.append(froerw)
                AmpCh.append(ampch)
                err_lab.append(y[triali])
                
                nrModes = psi.shape[1]
                wind = np.repeat(win, nrModes).reshape(nrModes,1).T
                trial = np.repeat(triali,nrModes).reshape(nrModes,1).T
                lab = np.repeat(y[triali],nrModes).reshape(nrModes,1).T
                lam = lam.reshape(1,len(lam))
                mu = ((np.log(lam)/self.info['dt']).imag)/(2*np.pi)
                #gamma = (np.log(lam)).real/self.info['dt'] #growth/decay rate
                
                Window.append(wind)
                Label.append(lab)
                Trial.append(trial)
                Lambda.append(lam)
                Mu.append(mu)
                Psi.append(psi[:n_chan,:])
                Amp.append(amp[:n_chan,:])

       
        #create DataFrame with results
        Amp = np.concatenate(Amp, axis = 1).T
        Psi = np.concatenate(Psi, axis = 1).T
        Lambda = np.concatenate(Lambda, axis = 1).T
        Mu = np.concatenate(Mu, axis = 1).T
        Window = np.concatenate(Window, axis = 1).T
        Trial= np.concatenate(Trial, axis = 1).T
        Label= np.concatenate(Label, axis = 1).T
        
        cols = ['PSI_'+item for item in channels]+['AMP_'+item for item in channels]
        data = np.c_[Psi, Amp]
        df = pd.DataFrame(columns = cols, data = data)
        df['Lambda'] = Lambda
        df['Mu'] = Mu
        df['win'] = Window
        df['trial'] = Trial
        df['label'] = Label
        self.results = df
        
        #create Dtaframe with Amplitude + Error 
        AppErr = np.r_[AppErr]
        AmpCh = np.r_[AmpCh]
        FroErW = np.asarray(FroErW)
        
        cols = ['AmpCh_'+item for item in channels]+['AppErr_'+item for item in channels]
        data = np.c_[AmpCh, AppErr]
        df2 = pd.DataFrame(columns = cols, data = data)
        df2['FroErW'] = FroErW
        df2['label'] = err_lab
        self.AmpCh_Err = df2
        
        sys.stdout.write('\n')
        return(self)
    
    
    def mode_stats(self, fbands, labels = [], unit_length = False, mode = 'PSI'):
        '''
        calculates descriptive stats of modes 

        Parameters
        ----------
        fbands: list, shape [[a,b],[c,d]]
            list of 2 element list with a,b,c,d are floats defining 
            frequency bounds of interest
        unit_length: bool, optional
            defines if modes should scaled to unit length. The default is False)
        labels : list, optional
            list containing int with class / trial labels 
        
        Returns
        -------
        Stats: dataframe 
            Dataframe containing stats for defined frequency

        '''
        Stats = {}
        if not labels:
            labels = list(set(self.y))
        
        for band in fbands:
            df = self._bands(band[0],band[1],labels)
            
            cols = [col for col in df.columns if mode in col]
            df = df[cols].abs()
        
            if unit_length == True: 
                normfact = np.sqrt(np.square(df).sum(axis=1))
                df = df.divide(normfact, axis=0)
            
            Stats[str(band[0]) + '-' + str(band[1])] = df.describe()
        
        self.Stats = Stats
        
        return(Stats)
    
##### plotting functions 

    def plot_statsCH(self):
        '''
        plot Channel statistics based on DMD.mode_stats() 
        self.stats shoul be present 

        Returns
        -------
        fig object 

        '''
        stats = self.Stats    
        channels = self.info['channels']
        mean_band = []
        median_band = []
        q1 = []
        q2 = []
        fbands = list(stats.keys())
        for band in stats.keys():
            mean_band.append(stats[band].loc['mean'].values)
            median_band.append(stats[band].loc['50%'].values)
            q1.append(stats[band].loc['25%'].values)
            q2.append(stats[band].loc['75%'].values)
            
        mean = np.c_[mean_band]
        median = np.c_[median_band]
        q1 = np.c_[q1]
        q2 = np.c_[q2]
            
        fig, ax = plt.subplots(2,2, figsize = (25,12)) 
        ax1 = sns.heatmap(mean, ax = ax[0,0], cmap = "YlOrBr")
        ax2 = sns.heatmap(median,ax = ax[0,1], cmap = "YlOrBr")
        ax3 = sns.heatmap(q1, ax = ax[1,0], cmap = "YlOrBr")
        ax4 = sns.heatmap(q2, ax = ax[1,1], cmap = "YlOrBr")
        
        _ = ax1.set_xticklabels(channels, rotation = 90, size = 6)
        _ = ax2.set_xticklabels(channels, rotation = 90, size = 6)
        _ = ax3.set_xticklabels(channels, rotation = 90, size = 6)
        _ = ax4.set_xlabel('Channels')
        _ = ax4.set_xticklabels(channels, rotation = 90, size = 6)
         
        _ =  ax1.set_ylabel('Mean')
        _ =  ax1.set_yticklabels(fbands,size = 6)
        _ =  ax2.set_ylabel('Median')
        _ =  ax2.set_yticklabels(fbands ,size = 6)
        _ =  ax3.set_ylabel('Q1')
        _ =  ax3.set_yticklabels(fbands ,size = 6)
        _ =  ax4.set_ylabel('Q2')
        _ =  ax4.set_yticklabels(fbands ,size = 6)
       
        return(fig)
      
    def plot_ChAmpErr(self, labels = []):
        '''
        plot Channel Amplitude and Error of Approximation based on DMD modes and 
        lambda  
        
        Parameters
        ----------
        labels : list, optional
            list containing int with class / trial labels 

        Returns
        -------
        fig object

        '''
        
        if not labels:
            labels = list(set(self.y))
        df = self.AmpCh_Err
        df = df[df['label'].isin(labels)]   
        
        channels = list(self.info['channels'])
        channels.append('labels')
        sns.set_style('whitegrid')
        fig, ax = plt.subplots(2,2, figsize = (25,12))
        
        #subplot1:AMPCH
        a = [col for col in df.columns if 'AmpCh' in col]
        a.append('label')
        ampch = df[a]
        ampch.columns = channels
        ampch1 = pd.melt(ampch, 'labels', var_name = 'chan')
        ax1 = sns.stripplot(y="value", x="chan", data = ampch1,
                      dodge=True, alpha=.25, zorder=1, color= 'g', ax = ax[0,0])
        _ = ax1.set_ylabel('max(EEG amp.)/ window')
        _ = ax1.set_xticklabels(channels[:-1],rotation = 90, size = 6)
        
        #subplot2: FroErW
        ax2 = sns.lineplot(x = df.index, y = 'FroErW', data = df, color = 'b', ax = ax[0,1])
        mean = np.repeat(df['FroErW'].mean(),len(df))
        df['mean'] = mean
        sns.lineplot(x = df.index , y = 'mean', data = df, color = 'r', ax = ax[0,1])
        _ = ax2.set_xlabel('Time window')
        
        #subplot3: Ampch heatmap
        ampch2 = np.array(ampch.values)
        ax3 = sns.heatmap(ampch2[:,:-1], ax = ax[1,0], cmap = "YlOrBr")
        _ = ax3.set_xticklabels(channels[:-1],rotation = 90, size = 6)
        _ = ax3.set_ylabel('max(EEG amp.)/ window')
        
        #subplot4: Error heatmap
        b = [col for col in df.columns if 'Err' in col]
        b.append('label')
        err = df[b]
        err.columns = channels
        err = np.array(err.values)
        ax4 = sns.heatmap(err[:,:-1], ax = ax[1,1], cmap = "YlOrBr")
        _ = ax4.set_xticklabels(channels[:-1],rotation = 90, size = 6)
        _ = ax4.set_ylabel("|X-X|'/ window")
        
        return(fig)
    
    def plot_frRPsi(self, labels = []): 
        '''
        plot DMD spectrum for each label 
        
        Parameters
        ----------
        labels : list, optional
            list containing int with class / trial labels 

        Returns
        -------
        fig object

        '''
        
        if not labels:
            labels = list(set(self.y))
          
        channels = list(self.info['channels'])
        channels.append('label')    
        channels.append('Mu')
        
        df = self.results
        df = df[(df['Mu'] > 0)].abs()
        df = df[df['label'].isin(labels)]
        
        a = [col for col in df.columns if 'PSI' in col]
        a.append('label')
        a.append('Mu')
        psi= df[a].abs()
        psi.columns = channels
        psi['power'] = (np.square(psi.loc[:,channels[:-2]]).sum(axis=1))
        psi['zero'] = np.repeat(0, len(psi))
       
        g = sns.FacetGrid(psi, col="label",col_wrap=2, height=8, aspect=.5)
        g.map(sns.scatterplot, 'Mu', 'power', alpha=.5, color = 'k' ) 
        g.map(plt.vlines, 'Mu', 'zero', 'power',alpha=.8, color = 'k')
        
        g.set_ylabels('|$\Phi$|\u00b2')
        g.set_xlabels('Frequency (Hz)')
        g.set(xlim=(0), ylim=(0))
        return(g)
        
    def plot_frRLam(self, labels = []):  
        '''
        plot lambda for each label and freq.  
        
        Parameters
        ----------
        labels : list, optional
            list containing int with class / trial labels 

        Returns
        -------
        fig object

        '''        
        df = self.results
        
        if not labels:
            labels = list(set(self.y))
            
        df = df[df['label'].isin(labels)]  
        df = df[(df['Mu'] > 0)].abs()
        df = df[['Mu','Lambda','label']]
        
        g = sns.FacetGrid(df, col="label", height=8, aspect=.5)
        g.map(sns.scatterplot, 'Mu', 'Lambda', alpha=.5, color = 'g') 
        
        g.set_ylabels('|$\lambda$|', size =12)
        g.set_xlabels('Frequency (Hz)')
        g.set(xlim=(0), ylim=(0))
        return(g)
   
    def get_PSI(self, fband, labels = [], unit_length = False):
        '''
        returns mode magnitude for frequency bands and labels defined 
        in fbands and labels 
        
        Parameters
        ----------
        fbands: list, shape [[a,b],[c,d]]
            list of 2 element list with a,b,c,d are floats defining 
            frequency bounds of interest
        unit_length: bool, optional
            dfines if modes should scaled to unit length. The default is False)
        labels : list, optional
            list containing int with class / trial labels 

        Returns
        -------
        PSI: dataframe 
            df with mode magnitude for frequency bands and labels defined 
            in fbands and labels
        '''
        
        PSI = _get_data(self, fband, 'PSI', unit_length, labels)
       
        return(PSI)
    
    def get_AMP(self, fband, labels = [], unit_length = False):
        '''
        returns mode amplitude for frequency bands and labels defined 
        in fbands and labels 
        
        Parameters
        ----------
        fbands: list, shape [[a,b],[c,d]]
            list of 2 element list with a,b,c,d are floats defining 
            frequency bounds of interest
        unit_length: bool, optional
            dfines if modes should scaled to unit length. The default is False)
        labels : list, optional
            list containing int with class / trial labels 

        Returns
        -------
        AMP: dataframe 
            df with mode amplitude for frequency bands and labels defined 
            in fbands and labels
        '''         
        AMP = _get_data(self, fband, 'AMP', unit_length, labels)
        
        return(AMP)
    
    def select_trials(self, selector, return_copy = True):
        '''
        Parameters
        ----------
        selector : list, 
            list of which trials to select 
        return_copy: bool, 
            default true, wheater to return copy or original dataframe
        Returns
        -------
        DMDobj
        Object with selected trials
        '''
        
        dmd_cp = cp.deepcopy(self)
        split = dmd_cp.results
        split = [split[split['trial'] == t] for t in selector]
        
        if return_copy == False:
            self.results = pd.concat(split)
            return(self.results) 
        else: 
            dmd_cp.results = pd.concat(split)
            return(dmd_cp)
        
##************************************** Functions ********************************************
##**************************************           ********************************************
    def _scale_input(self):
        """
        function to scale input to DMD based on definition in class 
        centre: zero mean 
        centre_norm: z-scoring 
        norm: amplitude normalisation
        
        Returns
        -------
        self
        """
       
        datascale = self.info['datascale']
        n_channels = self.info['n_chan']
        n_trials = self.info['trials']
        l_trials = self.info['trials_pts']
        X = self.X
        X = X.transpose(1,0,2).reshape(n_channels, -1)
         
        
        if datascale == 'centre':
            X -= np.mean(X, axis = 1).reshape(len(X),1)
    
        elif datascale == 'centre_norm':
            X -= np.mean(X, axis = 1).reshape(len(X),1)
            X /= np.std(X, axis = 1, ddof =1).reshape(len(X),1) #equivavalent to MATLAB std with Bessel's correction
            
        elif datascale == 'norm':
            X /= np.std(X, axis = 1, ddof =1).reshape(len(X),1) #equivavalent to MATLAB std with Bessel's correction
        
        if len(self.X.shape) == 3:
            X = X.reshape(n_channels, n_trials, l_trials).transpose(1,0,2)
        
        self.X = X
        self.scaled = True 
        
        return (self)    




    def _bands(self, lower_bound, upper_bound, labels = []):
        '''
        find modes of defined frequency band
    
        Parameters
        ----------
        lower_bound : int, float 
            != 0 lower searchbound 
        upper_bound : int, float 
            lower searchbound 
        labels : list, optional
            list containing int with class / trial labels 
    
        Returns
        -------
        df_bands: dataframe, 
            filtered dataframe containing modes in defined frequencie bands
    
        '''
        # make sure labels is a list! 
        df = self.results
        
        if not labels:
            labels = list(set(df.label))
            
        df = df[df['label'].isin(labels)]
        
        if lower_bound == 0: 
            lower_bound += .00000001 #exclude 0 frequency modes
            
        df_bands = df[(df['Mu'] >= lower_bound) & (df['Mu'] < upper_bound)]
        
        return(df_bands)


           
def _get_data(self, fband, mode, unit_length, labels = []):
    '''
    Parameters
    ----------
    fbands: list, shape [[a,b],[c,d]]
        list of 2 element list with a,b,c,d are floats defining 
    mode : str, {'AMP','PSI'} 
        defnes what to get Amplitude 'AMP' or Magnitude 'PSI'  
    labels : list, optional
        list containing int with class / trial labels 

    Returns
    -------
    data: Dataframe 
        Dataframe with desired columns and rows 
    '''
    
    df = self._bands(fband[0],fband[1], labels)
    cols = [col for col in df.columns if mode in col]
    data = cp.deepcopy(df[cols])
    if unit_length == True: 
        normfact = np.sqrt(np.square(data).sum(axis=1))
        data = data.divide(normfact, axis=0)
    
    data['Mu'] = df.Mu
    data['win'] = df.win
    data['trial'] = df.trial
    data['label'] = df.label

    return(data)
        
def _X_aug_h(X,h):
    """
    Shift-stack data matrix with stacking factor h 
    
    Parameters
    ----------
    X : data_matrix of array_like 
    
    h : int, Stackingfactor
    
    Returns
    -------
    Xaug: Shift-stacked data matrix of array_like
    
    """
    m = X.shape[1]+1
    n = X.shape[0]
    Xaug = np.zeros((h*n,m-h))
    Xvec = np.concatenate(X.T, axis=0)
    
    for i in range(Xaug.shape[1]):
        Xaug[:,i] = Xvec[i*n:h*n+i*n]
        
    return(Xaug)
    
    
def _DMD_comp(XY, h, algorithm, truncation = False):
    
    """
    Algorithm to compute the DMD basis of a data matrix XY \in R^(n,m), where
    n is the dimension of the dynamical system and m is the number of
    snapshots
    
    Parameters
    ----------
    XY : array, shape (n,m)
        the data to compute DMD 
    h: int 
        stacking factor for shift stacking 
    truncation: dict 
        defines wheater to truncate with hard thresholding
        
     
    Returns
    -------
        Psi
        lam 
        Xhat
        Xaug
        condXaug 
        z0  
        Amp
     
    """
   
    X = XY[:,:-1]
    Y = XY[:,1:]
   
    Xaug = matrix(_X_aug_h(X,h))
    Yaug = matrix(_X_aug_h(Y,h))
    
    condXaug = cond(Xaug)
    
    U,S,V = svd(Xaug, full_matrices=False, lapack_driver = 'gesvd') #'gesvd' general rectangular approach used by MATLAB
    U = matrix(U)
    V = matrix(V.T)
    
    if truncation['method'] != None:
        
        # define how many modes to keep
        if truncation['method'] == 'optht':
            r = optht(Xaug, sv=S, sigma=None)
        elif truncation['method'] == 'cut':
            assert truncation['keep'] != None, 'Please specify how many SVD modes to keep when truncation is set to cut!'
            r = truncation['keep']

        #keep r modes     
        U = U[:,:r]
        S = S[:r]
        V = V[:,:r]
     
    S1 =  matrix(diag(S**-1))
    S0 = matrix(diag(S**-0.5))
    S2 = matrix(diag(S**0.5))
    A = S0 * ((U.T*Yaug) * (V*S1)) * S2
    
    [lam,W_hat] = eig(A)
    W = S2 * W_hat
    
    #Standard
    if algorithm == 'standard':
        Psi = U * W
    
    #exact DMD 
    if algorithm == 'exact':
       Psi = Yaug * (V * (S1 * W))
    
    z0 = pinv(Psi) * Xaug[:,0] 
    Xhat = np.zeros((Psi.shape[0],XY.shape[1]), dtype = 'complex_');
    Xhat[:,0] = np.ravel(Xaug[:,0])
    for i in range(1,XY.shape[1]):
        Xhat[:,i] = np.ravel(Psi*diag(lam**i)*z0)
    
    Amp = Psi * diag(np.ravel(z0))
    
    return(Psi, lam, Xhat, Xaug, condXaug, z0, Amp)   
    

def _check_option(parameter, value, allowed_values, extra=''):
    """Check the value of a parameter against a list of valid options.

    Raises a ValueError with a readable error message if the value was invalid.

    Parameters
    ----------
    parameter : str
        The name of the parameter to check. This is used in the error message.
    value : any type
        The value of the parameter to check.
    allowed_values : list
        The list of allowed values for the parameter.
    extra : str
        Extra string to append to the invalid value sentence, e.g.
        "when using ico mode".

    Raises
    ------
    ValueError
        When the value of the parameter was not one of the valid options.
    """
    if value in allowed_values:
        return True

    # Prepare error message
    extra = ' ' + extra if extra else extra
    msg = ("Invalid value for the '{parameter}' parameter{extra}. "
           '{options}, but got {value!r} instead.')
    if len(allowed_values) == 1:
        options = 'The only allowed value is %r' % allowed_values[0]
    else:
        options = 'Allowed values are '
        options += ', '.join(['%r' % v for v in allowed_values[:-1]])
        options += ' and %r' % allowed_values[-1]
    raise ValueError(msg.format(parameter=parameter, options=options,
                                value=value, extra=extra))
    
    
    
