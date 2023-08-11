import numpy as np
import neo
import scipy
import scipy.io as sio
import pandas as pd
from scipy import signal
from fooof import FOOOF

#from externals.SpectralEvents import spectralevents_functions as tse

def load_ns6_analog(fpath, downsample_rate, from_ns6=True, save=False, channel_step=1):

    # Load LFP data directly from NS5
    if from_ns6:
        stream_index = 1
        ns6 = neo.BlackrockIO(fpath + '.ns6')

        num_channels = ns6.signal_channels_count(stream_index=stream_index)
        lfp_channels = list(range(num_channels))

        # Loop over each channel to avoid blowing up RAM, really slow...
        lfp_data_list = list()
        for idx_start in range(0, num_channels, channel_step):
            if idx_start + channel_step < num_channels:
                channel_indexes = tuple(lfp_channels[idx_start:idx_start+channel_step])
            else:
                channel_indexes = tuple(lfp_channels[idx_start:])   

            print(channel_indexes, end=' ')
            channel_data = ns6.get_analogsignal_chunk(
                stream_index=stream_index, channel_indexes=[channel_indexes]).squeeze().transpose()
                
            channel_data = channel_data[:, ::downsample_rate]
            lfp_data_list.append(channel_data)

            lfp_data = np.concatenate(lfp_data_list)
            tstart, tstop = ns6._seg_t_starts, ns6._seg_t_stops
            lfp_times = np.linspace(tstart, tstop, lfp_data.shape[1]).squeeze()
                
            if save:
                np.save(fpath + f'_lfp_channels_{downsample_rate}x_downsample.npy', lfp_data)
                np.save(fpath + f'_lfp_times_{downsample_rate}x_downsample.npy', lfp_times)

    else:
        lfp_data = np.load(fpath+ f'_lfp_channels_{downsample_rate}x_downsample.npy')
        lfp_times = np.load(fpath+ f'_lfp_times_{downsample_rate}x_downsample.npy')

    return lfp_data, lfp_times

def get_aperiodic(dpl, fs, min_freq=3, max_freq=80, nperseg=10_000, noverlap=5000):
    fm = FOOOF()
    freqs, Pxx = signal.welch(dpl, fs, nperseg=nperseg, average='median', noverlap=noverlap)
    
    freq_range = [min_freq, max_freq]
    # Define frequency range across which to model the spectrum
    fm.report(freqs, Pxx, freq_range)

    aperiodic_params = fm.get_results().aperiodic_params
    offset, exponent = aperiodic_params[0], aperiodic_params[1]
    
    return offset, exponent

def bandpower(x, fs, fmin, fmax):
    f, Pxx = scipy.signal.periodogram(x, fs=fs)
    ind_min = np.argmax(f > fmin) - 1
    ind_max = np.argmax(f > fmax) - 1
    return np.trapz(Pxx[ind_min: ind_max], f[ind_min: ind_max])

# Bands freq citation: https://www.frontiersin.org/articles/10.3389/fnhum.2020.00089/full
def get_dataset_bandpower(x, fs):
    freq_band_list = [(0,13), (13,30), (30,50), (50,80)]
    
    x_bandpower_list = list()
    for idx in range(x.shape[0]):
        x_bandpower = np.array([bandpower(x[idx,:], fs, freq_band[0], freq_band[1]) for freq_band in freq_band_list])
        x_bandpower_list.append(x_bandpower)
        
    return np.vstack(x_bandpower_list)

def get_dataset_psd(x_raw, fs, return_freq=True, max_freq=200, nperseg=500, noverlap=50):
    """Calculate PSD on observed time series (rows of array)"""
    x_psd = list()
    for idx in range(x_raw.shape[0]):
        f, Pxx = signal.welch(x_raw[idx,:], fs, nperseg=nperseg, average='median', noverlap=noverlap)
        x_psd.append(Pxx[(f<max_freq)&(f>0)])
    if return_freq:
        return np.vstack(np.log(x_psd)), f[(f<max_freq)&(f>0)]
    else:
        return np.vstack(np.log(x_psd))
    
# Source: https://stackoverflow.com/questions/44547669/python-numpy-equivalent-of-bandpower-from-matlab
def bandpower(x, fs, fmin, fmax):
    f, Pxx = scipy.signal.periodogram(x, fs=fs)
    ind_min = scipy.argmax(f > fmin) - 1
    ind_max = scipy.argmax(f > fmax) - 1
    return scipy.trapz(Pxx[ind_min: ind_max], f[ind_min: ind_max])

def plot_avg_psd(Pxx_array, f, ax, label='PSD', max_freq=100, color=f'C0'):
    f_mask = f < max_freq
    f = f[f_mask]
    Pxx_array = Pxx_array[:, f_mask]
    mean_Pxx = np.mean(Pxx_array, axis=0)
    std_Pxx = np.std(Pxx_array, axis=0) / np.sqrt(Pxx_array.shape[0])

    ax.plot(f, mean_Pxx, color=color, label=label)

    ax.fill_between(f, mean_Pxx - std_Pxx, mean_Pxx + std_Pxx, color=color, alpha=0.3)
    ax.set_xlabel('Frequency')
    ax.set_yscale('log')

    return ax


def get_phase_maps(lfp_data_array, lfp_beta_phase, lfp_beta_envelope, k=2):
    num_trials, _, num_times = lfp_beta_phase.shape
    
    channel_filter = emap_df['label_idx'].values
    noise_mask = ~np.isin(channel_filter, noise_channels) 

    row_filter = emap_df['row'].values - np.min(emap_df['row'].values)
    col_filter = emap_df['col'].values - np.min(emap_df['col'].values)
    num_rows, num_cols = np.max(row_filter) + 1, np.max(col_filter) + 1

    phase_array = np.full((num_rows, num_cols, num_trials, num_times), np.nan)
    amplitude_map = np.full((num_rows, num_cols, num_trials, num_times), np.nan)
    voltage_map = np.full((num_rows, num_cols, num_trials, num_times), np.nan)
    electrode_map = np.full((num_rows, num_cols), np.nan)
    
    row_noise, col_noise = row_filter[~noise_mask], col_filter[~noise_mask]
    row_filter, col_filter, channel_filter = row_filter[noise_mask], col_filter[noise_mask], channel_filter[noise_mask]

    for row_idx, col_idx, channel_idx in zip(row_filter, col_filter, channel_filter):
        # Shape into row x col x trial x times
        phase_array[row_idx, col_idx, :, :] = lfp_beta_phase[:, channel_idx, :]
        amplitude_map[row_idx, col_idx, :, :] = lfp_beta_envelope[:, channel_idx, :]
        voltage_map[row_idx, col_idx, :, :] = lfp_data_array[:, channel_idx, :]

        electrode_map[row_idx, col_idx] = channel_idx
        
    # Fill with average noise 
    phase_array = get_neigborhood_average(phase_array, row_noise, col_noise, k=1)
    
    # Phase gradient
    # =============================================================================
    phase_gradient_array = phase_gradient_vectorized(np.abs(phase_array))
    # phase_gradient_array = phase_gradient_vectorized(phase_array)
    
    # Phase speed
    # =============================================================================
    phase_speed_array = phase_speed_vectorized(phase_gradient_array)
    
    # Phase directionality
    # =============================================================================
    phase_directionality = phase_gradient_array / np.abs(phase_gradient_array)

    # Gradient coherence and contiuity
    # =============================================================================
    gradient_coherence, gradient_continuity = gradient_coherence_continuity_vectorized(phase_directionality)
    global_coherence = np.mean(gradient_coherence, axis=(0,1))
    global_continuity = np.mean(gradient_continuity, axis=(0,1))

    # Circular variance of phases (sigma_p)       
    # =============================================================================   
    sigma_p = sigma_p_vectorized(phase_array)

    # Circular variance of phase directionality (sigma_g)       
    # =============================================================================
    global_directionality = np.mean(phase_directionality, axis=(0,1))
    sigma_g = np.abs(global_directionality)
    

    # Synchronized wave and planar wave
    # The threshold for the circulars phase and phase gradient to distinguish planar waves and synchronized waves
    judge_theta=[0.85,0.5]
    syn=((np.array(sigma_p) >judge_theta[0]) & (np.array(sigma_g)<=judge_theta[1]))
    planar=(np.array(sigma_g)>judge_theta[1])

    cps = np.array([count_critical(phase_gradient_array[:,:,trial_idx,:].transpose(2,0,1)) for
                    trial_idx in range(num_trials)])
    
    nclockwise, nanticlockwise, nmaxima, nminima = cps[:,0,:], cps[:,1,:], cps[:,2,:], cps[:,3,:]
    clockwise = nclockwise + nanticlockwise
    peaks = nmaxima + nminima
    radial = (peaks==1) & (clockwise==0)  & (~planar) & (~syn)
    
    all_wave_kind_list = list()
    for trial_idx in range(num_trials):
        wave_kind = syn[trial_idx,:], planar[trial_idx,:], radial[trial_idx,:]
        # Remove too short time 
        duration_threshold=5 
        effective_wave =[]  
        for wave_idx, wave in enumerate(wave_kind):
            effective_wave.append(function_remove_short(wave, duration_threshold))
            
        unclass = ~((np.sum(effective_wave,axis=0))>0 )
        syn_trial, planar_trial, radial_trial = effective_wave 
        all_wave_kind = syn_trial, planar_trial, radial_trial, (unclass+0)
        all_wave_kind_list.append(all_wave_kind)
    all_wave_kind_array = np.array(all_wave_kind_list)


    mask = ~np.isnan(phase_array)
    

    phase_map_dict = {
        'phase': phase_array, 'phase_speed': phase_speed_array,
        'phase_gradient': phase_gradient_array, 'num_peaks': peaks,
        'phase_directionality': phase_directionality, 'gradient_continuity': gradient_continuity,
        'global_continuity': global_continuity, 'global_coherence': global_coherence,
        'global_directionality': global_directionality,
        'gradient_coherence': gradient_coherence, 'sigma_p': sigma_p,
        'sigma_g': sigma_g, 'syn': syn, 'planar': planar, 'cps': cps,
        'clockwise': clockwise, 'radial': radial, 'all_wave_kind': all_wave_kind_array,
        'amplitude_map': amplitude_map, 'electrode_map': electrode_map,
        'voltage_map': voltage_map, 'mask': mask, 'channel_filter': channel_filter}

    return phase_map_dict

def phase_gradient_vectorized(phase, k=2):
    n_rows, n_cols, n_trials, n_times = phase.shape
    phase_gradient=np.zeros((n_rows,n_cols,n_trials,n_times),dtype=complex)
    k_indices = list(range(-k, k+1))
    k_indices.remove(0)
    
    for x in range(n_rows)[:]:
        for y in range(n_cols)[:]:
            add_x = np.zeros((n_trials, n_times), dtype=complex)
            add_number_x = 0
            add_y = np.zeros((n_trials, n_times), dtype=complex)
            add_number_y = 0
            for row_r in k_indices:
                if((n_rows-1)>=x+row_r>=0):
                    if(row_r<0): alpha=0
                    else:alpha=np.pi
                    ele_i= x+row_r 
                    # print(ele_i)
                    add_x+=(((( (phase[ele_i,y,:,:] - phase[x,y,:,:])+np.pi)%(2*np.pi)-np.pi)/np.abs(row_r))*cmath.exp(1j*alpha))
                    add_number_x+=1
                    
            for col_r in k_indices:
                if((n_cols-1)>=y+col_r>=0):
                    if(col_r<0):alpha=0.5*np.pi
                    else:alpha=1.5*np.pi
                    ele_j= y+col_r 
                    # print(ele_j)
                    add_y+=(((( (phase[x,ele_j,:,:] - phase[x,y,:,:])+np.pi)%(2*np.pi)-np.pi)/np.abs(col_r))*cmath.exp(1j*alpha))
                    add_number_y+=1
            phase_gradient[x,y,:,:]=(add_x/add_number_x + add_y/add_number_y)
    return phase_gradient

def phase_speed_vectorized(phase_gradient_array, fre_m=21.5):
    '''
    Calculate the speed (see Methods in the paper). 

    Parameters
    ----------
    fre_m : float
         The mean frequency of the respective beta bands.
        
    Returns
    -------
    list
        The phase speed.  
    '''
    electrode_spacing=0.4  #Spacing between electrodes for the Utah arrays (mm) 
    phase_speed=(2 * np.pi * fre_m / (np.nanmean(np.abs(phase_gradient_array), axis=(0,1)) * electrode_spacing * 10e-2))
    return phase_speed

def gradient_coherence_continuity_vectorized(phase_directionality, k=2):
    '''
    Calculate the gradient coherence and continuity. 

    Parameters
    ----------
    phase_directionality : np.array [10]*[10]
        ND numpy array of phase.
        
    Returns
    -------
    np.array
        The gradient coherence.  
    '''
    n_rows, n_cols, n_trials, n_times = phase_directionality.shape
    k_indices = list(range(-k, k+1))
    k_indices.remove(0)

    gradient_coherence = np.zeros((n_rows,n_cols, n_trials, n_times),dtype=complex)
    gradient_continuity = np.zeros((n_rows,n_cols, n_trials, n_times),dtype=float)
    
    for x in range(n_rows) :
        for y in range(n_cols) :
            phase_d_add = np.zeros((n_trials, n_times), dtype=complex)
            grad_c_add = np.zeros((n_trials, n_times), dtype=float)
            
            add_number = 0 # Denominator for local avergage
            
            # Sweep over k neigborhood around (x,y)
            for row_r in k_indices:
                for col_r in k_indices:
                    if((n_rows-1>=x+row_r>=0)&(n_cols-1>=y+col_r>=0)):
                        ele_i= x+row_r
                        ele_j= y+col_r
                        local_phase_d = phase_directionality[ele_i,ele_j,:,:] 
                        phase_d_add += local_phase_d
                        
                        # Complex dot product to calculate gradientcontinuity
                        real_prod = np.real(local_phase_d) * np.real(phase_directionality[x,y,:,:])
                        imag_prod = np.imag(local_phase_d) * np.imag(phase_directionality[x,y,:,:])
                        s = real_prod + imag_prod
                        
                        grad_c_add += s
                        add_number += 1
            
            gradient_coherence[x,y,:,:] = phase_d_add / add_number
            gradient_continuity[x,y,:,:] = grad_c_add / add_number
            
    return gradient_coherence, gradient_continuity

def count_critical(phase_gradient):
    '''
    Find critical points in the phase gradient map.

    Parameters
    ----------
    phase_gradient_list : np.array [T]*[10*10]
    The list of the phase gradient.
        
    Returns
    -------
    nclockwise : np.array
        The number of clockwise centers found at each time point.
    nanticlockwise : np.array
        The number of anticlockwise centers found at each time point.
    nsaddles : np.array
        The number of saddle points.
    nmaxima : np.array
        The number of local maxima.
    nminima : np.array
        The number of local minima.
    '''
    data =  phase_gradient
    
    # curl  
    curl = np.complex64([[-1-1j,-1+1j],[1-1j,1+1j]])
    curl = convolve2d(curl,np.ones((2,2))/4,'full')
    winding = np.array([convolve2d(z,curl,'same','symm').real for z in data])

    # cortical points
    ok        = ~(np.abs(winding)<1e-1)[...,:-1,:-1]
    ddx       = np.diff(np.sign(data.real),1,1)[...,:,:-1]/2
    ddy       = np.diff(np.sign(data.imag),1,2)[...,:-1,:]/2
    saddles   = (ddx*ddy==-1)*ok
    maxima    = (ddx*ddy== 1)*(ddx==-1)*ok
    minima    = (ddx*ddy== 1)*(ddx== 1)*ok
    sum2 = lambda x: np.sum(np.int32(x),axis=(1,2))
    nclockwise = sum2(winding>3)
    nanticlockwise = sum2(winding<-3)
    nsaddles   = sum2(saddles  )
    nmaxima    = sum2(maxima   )
    nminima    = sum2(minima   )
    return nclockwise, nanticlockwise, nmaxima, nminima

def get_phase_gradient_arrows(direction_map, trial_idx, time_idx):
    idx_flat = np.array(list(np.ndindex(direction_map[:,:,trial_idx,time_idx].shape)))

    U = np.array((np.nditer(np.imag(direction_map[:,:,trial_idx,time_idx]))))
    V = np.array((np.nditer(np.real(direction_map[:,:,trial_idx,time_idx]))))
    
    denom = np.sqrt(U**2 + V**2)
    U = U / denom
    V = V / denom
    
    return idx_flat[:,0], idx_flat[:,1], U, V

def sigma_p_vectorized(phase):
    '''
    Calculate the circular of the phase. 

    Parameters
    ----------
    phase : np.narry [10]*[10]
         ND numpy array of phase.
        
    Returns
    -------
    int
        Sigma_p.  
    '''
    n_rows, n_cols, _, _ = phase.shape
    
    func = np.vectorize(lambda x: cmath.exp(1j*x))
    out = func(phase)
    out = np.sum(out, axis=(0,1))
    sigma_p = np.abs(out / (n_rows * n_cols))
    return sigma_p

def function_remove_short(wave,cutoff):
    '''
    Remove the short duration (less than the cutoff) of the waves.
     
    Parameters
    ----------
    wave :  list (bool) 
        List of wave.
    cutoff :  int
        The threshold for the duration.
        
    Returns
    -------
    np.array
        The array of the effective wave. 
    '''
    a,b  = function_get_edges(wave)
    gaps = b-a
    keep = np.array([a,b])[:,gaps>cutoff]
    newgaps = function_set_edges(keep.T,len(wave))
    return newgaps

def function_get_edges(wave):
    '''
    Find the starts and the ends of the wave.
     
    Parameters
    ----------
    wave : list (bool)
        List of waves.
    Returns
    -------
    np.array
        The array of starts and ends.  
    '''
 
    if len(wave)<1:
        return np.array([[],[]])
    starts =  np.where(np.diff(np.int32(wave))==1)
    stops  =  np.where(np.diff(np.int32(wave))==-1)
    if wave[0]: 
        starts=np.insert(starts,0,int(0))
    if wave[-1]: 
        stops=np.insert(stops,int(len(stops[0])),int(len(wave))) 
       
    if (isinstance(stops,tuple)):  
            stops=np.array(stops[0])
    if (isinstance(starts,tuple)):
            starts=np.array(starts[0])
    return np.array([starts+1, stops+1])

def function_set_edges(edges,L):
    '''
    Set the starts and the ends of the wave.
     
    Parameters
    ----------
    edges :  np.array 
        The array of the starts and the ends of the waves.
    L :  int
        The length of the wave.
        
    Returns
    -------
    np.array
        The array of wave.  
    '''
    x = np.zeros(shape=(L,),dtype=np.int32)
    for (a,b) in edges:
        x[a:b]= 1
    return x

def get_neigborhood_average(phase_array, row_noise, col_noise, k=1):
    n_rows, n_cols, n_trials, n_times = phase_array.shape
    
    for x, y in zip(row_noise, col_noise):
        assert np.all(np.isnan(phase_array[x,y,:,:]))

        neighborhood_signal = list()

        k_indices = list(range(-k, k+1))
        k_indices.remove(0)
        for row_r in k_indices:
            for col_r in k_indices:
                if((n_rows-1>=x+row_r>=0)&(n_cols-1>=y+col_r>=0)):
                    ele_i= x+row_r
                    ele_j= y+col_r
                    neighborhood_signal.append(phase_array[ele_i, ele_j])
                    
        phase_array[x,y,:,:] = np.mean(np.stack(neighborhood_signal), axis=0)
        return phase_array
        

