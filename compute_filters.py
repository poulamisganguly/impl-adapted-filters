import numpy as np
import astra
import tomopy
import skimage
from scipy.signal import fftconvolve
import time
from tqdm.auto import tqdm, trange


def create_basis(Nd, Nl):
    """Creates a basis for filter expansion in Fourier space.

    Parameters:
        Nd (int): length of detector 
        Nl (int): number of large bins (with width 1.0)
    
    Returns:
        basisn (numpy.ndarray): filter basis
    
    """
    # exponential binning basis in real space (see MR-FBP)
    fs = Nd
    if fs%2==0:
        fs += 1
    mf = int(fs/2)

    w=1
    c=mf

    bas = np.zeros(fs,dtype=np.float32)
    basis = []
    count=0
    while c<fs:
        bas[:]=0
        ll = c
        rr = c+w
        if rr>fs: rr=fs
        bas[ll:rr]=1
        if ll!=0:
            ll = fs-c-w
            rr = ll+w
            if ll<0: ll=0
            bas[ll:rr]=1
        basis.append(bas.copy())
        c += w
        count += 1
        if count>Nl:
            w=2*w
    # Fourier transform of basis
    Nb = np.shape(basis)[0] # no of bins
    basisn = np.zeros([mf+1,Nb])
    for i in range(0,Nb):
        basisn[:,i] = np.fft.rfft(np.roll(basis[i],-basis[0].shape[0]//2+1)).real

    basisn[:,0] = 1.0

    # normalisation
    for i in range(Nl,Nb):
        basisn[:,i] = basisn[:,i]/2**(i-Nl)

    return basisn

def fbp_with_filter(data, angles, N, kernel, custom_filter, circle=False):
    """Computes FBP reconstruction with custom filter and backprojection routine
    
    Parameters:
        data (numpy.ndarray): projection data
        angles (numpy.ndarray): projection angles
        N (int): number of pixels in rows and columns of reconstruction
        kernel (str) : backprojection kernel (allowed: 'astra-strip', 'astra-line', 'astra-linear', 'astra-cuda', 'iradon')
        custom_filter (numpy.ndarray) : custom filter (in Fourier space)
        circle (bool) : mask circular region (default False, set True for filter computation)

    Returns:
        rec (numpy.ndarray): reconstruction with custom filter
        """
    # filter data
    S = np.fft.rfft(data)
    S_f = custom_filter*S
    Q = np.fft.irfft(S_f)

    # create backprojector
    if kernel == 'iradon':
        theta = np.rad2deg(angles)
        rec = skimage.transform.iradon(Q.T, theta, circle=True, filter=None, interpolation='cubic')
        rec = rec.astype(float)[N//2+1:-(N//2-1),N//2:-N//2] # adjust iradon shift
        if circle:
            astra.extrautils.clipCircle(rec)

    else:
        proj_type = kernel.split('-')[1] # extract proj type by removing 'astra-'
        proj_geom = astra.create_proj_geom('parallel', 1.0, data.shape[1], angles)
        vol_geom = astra.create_vol_geom(N, N)
        pid = astra.create_projector(proj_type, proj_geom, vol_geom)
        p = astra.OpTomo(pid)
        rec = (p.T*Q).reshape([N, N])
        if circle:
            astra.extrautils.clipCircle(rec)
    return rec

def gridrec_with_filter(data, angles, N, kernel, custom_filter, circle=False):
    """Computes Gridrec reconstruction with custom filter and backprojection routine
    
    Parameters:
        data (numpy.ndarray): projection data
        angles (numpy.ndarray): projection angles
        N (int): number of pixels in rows and columns of reconstruction
        kernel (str) : backprojection kernel (allowed: 'tomopy-gridrec')
        custom_filter (numpy.ndarray) : custom filter (in Fourier space)
        circle (bool) : mask circular region (default False, set True for filter computation)

    Returns:
        rec (numpy.ndarray): reconstruction with custom filter

    """
    # filter data
    S = np.fft.rfft(data)
    S_f = custom_filter*S
    Q = np.fft.irfft(S_f)

    # create backprojector
    if kernel == 'tomopy-gridrec':
        
        rec = tomopy.recon(np.expand_dims(Q,1), angles, algorithm='gridrec', filter_name='none',
                             sinogram_order=False, num_gridx=N, num_gridy=N)[0,:,:].astype(float)
        if circle:
            astra.extrautils.clipCircle(rec)
    return rec

class ComputedFilters:
    """Class of implementation-adapted filters
    
    Attributes:
        data (numpy.ndarray): sinogram data
        angs (numpy.ndarray): projection angles
        N (int)           : num of pixels in row and column of reconstruction (i.e. reconstruction is NxN)
        exp_binning (bool): set True to use exponential binning, default is False
        large_bins (int)  : number of bins of width 1 in exponential binning basis
    
    Methods:
        filter_fbp(impl): Computes the filter given an implementation
    """
    
    def __init__(self, data, angs, N, exp_binning=False, large_bins=2):
        self.data = data
        self.angles = angs
        self.N = N
        self.exp_binning = exp_binning
        self.Nl = large_bins
        
        # compute number of detector pixels and proj angles
        self.num_det_pix = self.data.shape[1]
        self.num_proj_ang = self.data.shape[0]
        
        # zero pad data
        self.padded = np.pad(self.data, ((0,), (0,)), 'constant')


    def filter_fbp(self, impl):
        """Computes implementation-adapted filter
        
        Parameters:
            impl (str)   : implementation of FBP or Gridrec
                        allowed values of impl are 'astra-strip', 'astra-line', 'astra-linear', 'astra-cuda', tomopy-gridrec' and 'iradon'. 
                        The forward projection kernel is always 'strip'
        
        Returns:
            filt (numpy.ndarray): implementation-adapted filter
        """
        # create basis
        if self.exp_binning:
            bas = create_basis(Nd=self.num_det_pix, Nl=self.Nl)
        else:
            bas = create_basis(Nd = self.num_det_pix, Nl=self.num_det_pix)
        
        # create algorithm matrix
        A = np.zeros([int(self.num_det_pix*self.num_proj_ang),bas.shape[1]])
        
        # compute A by running fbp for each basis component
        for ii in trange(A.shape[1], desc="Creating projected reco matrix"):
            if impl=='tomopy-gridrec':
                img = gridrec_with_filter(self.data, self.angles, self.N, impl, bas[:,ii], circle=True) 
            else:
                img = fbp_with_filter(self.data, self.angles, self.N, impl, bas[:,ii], circle=True)
            
            # create columns of A
            f_proj_geom = astra.create_proj_geom('parallel', 1.0, self.padded.shape[1], self.angles)
            f_vol_geom = astra.create_vol_geom(self.N, self.N)
            fpid = astra.create_projector('strip', f_proj_geom, f_vol_geom)
            fp = astra.OpTomo(fpid)
            A[:,ii] = fp*img

        # least squares fit
        h = np.linalg.lstsq(A,self.data.flatten(),rcond=None)[0]
        filt = bas.dot(h)
        
        return filt
    

class Reconstructions:
    """Class of reconstructions from analytical algorithms
    
    Attributes:
        sinogram (numpy.ndarray): sinogram data
        angs (numpy.ndarray): projection angles
        n (int)           : num of pixels in row and column of reconstruction (i.e. reconstruction is nxn)
        filter(str)       : filter (default 'shepp-logan', other allowed value: 'ram-lak')
    
    Methods:
        astra_fbp(proj_type): Computes FBP reconstruction using a proj_type (str) in the ASTRA toolbox
        astra_fbp_cuda(proj_type): Computes FBP reconstruction using a GPU projector in the ASTRA toolbox
        tomopy_gridrec(): Computes Gridrec reconstruction using TomoPy
        skimage_iradon(): Computes iradon reconstruction using scikit-image
    """
    def __init__(self, sinogram, n, angs, filter='shepp-logan', circle=True):
        self.sinogram = sinogram
        self.filter = filter
        self.n = n
        self.angs = angs
        self.circle = circle
    
    def astra_fbp(self, proj_type='strip'):
        """Computes ASTRA FBP reconstruction for a given projector type
        
        Parameters:
            proj_type (str): type of projector (e.g. 'strip' (default), 'line', 'linear',...)
        
        Returns:
            rec (numpy.ndarray): reconstruction
        """
        n_d = self.sinogram.shape[1]
        vol_geom = astra.create_vol_geom(self.n, self.n)
        proj_geom = astra.create_proj_geom('parallel', 1.0, n_d, self.angs)
        proj_id = astra.create_projector(proj_type, proj_geom, vol_geom)
        sino_id = astra.data2d.create('-sino', proj_geom, data=self.sinogram)
        
        rec_id = astra.data2d.create('-vol', vol_geom)
        cfg = astra.astra_dict('FBP')
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = sino_id
        cfg['ProjectorId'] = proj_id

        alg_id = astra.algorithm.create(cfg)
        cfg['option'] = {}
        cfg['option']['FilterType'] = self.filter
        
        astra.algorithm.run(alg_id, 1)

        rec = astra.data2d.get(rec_id)
        if self.circle:
            astra.extrautils.clipCircle(rec)
        
        return rec.astype(float)
    
    def astra_fbp_cuda(self, proj_type='cuda'):
        """Computes ASTRA FBP reconstruction for a given GPU projector type
        
        Parameters:
            proj_type (str): type of projector (e.g. 'cuda' (default), currently only 'cuda' is supported)
        
        Returns:
            rec (numpy.ndarray): reconstruction
        """
        n_d = self.sinogram.shape[1]
        vol_geom = astra.create_vol_geom(self.n, self.n)
        proj_geom = astra.create_proj_geom('parallel', 1.0, n_d, self.angs)
        proj_id = astra.create_projector(proj_type, proj_geom, vol_geom)
        sino_id = astra.data2d.create('-sino', proj_geom, data=self.sinogram)
        
        rec_id = astra.data2d.create('-vol', vol_geom)
        cfg = astra.astra_dict('FBP_CUDA')
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = sino_id

        alg_id = astra.algorithm.create(cfg)
        cfg['option'] = {}
        cfg['option']['FilterType'] = self.filter
        
        astra.algorithm.run(alg_id, 1)

        rec = astra.data2d.get(rec_id)
        if self.circle:
            astra.extrautils.clipCircle(rec)
        
        return rec.astype(float)
    
    def tomopy_gridrec(self):
        """Computes Gridrec reconstruction in TomoPy
        
        Returns:
            rec (numpy.ndarray): reconstruction
        """
        n_d = self.sinogram.shape[1]
        if self.filter=='ram-lak':
            filt_tomopy = 'ramlak'
        if self.filter=='shepp-logan':
            filt_tomopy = 'shepp'
        rec = tomopy.recon(np.expand_dims(self.sinogram,1), self.angs, algorithm='gridrec', filter_name=filt_tomopy,
                             sinogram_order=False, num_gridx=self.n, num_gridy=self.n)[0,:,:].astype(float)
        if self.circle:
            astra.extrautils.clipCircle(rec)
        return rec
    
    def skimage_iradon(self):
        """Computes iradon reconstruction in scikit-image
        
        Returns:
            rec (numpy.ndarray): reconstruction
        """
        if self.filter=='ram-lak':
            filt_iradon = 'ramp'
        if self.filter=='shepp-logan':
            filt_iradon = 'shepp-logan'
        theta = np.rad2deg(self.angs)
        rec = skimage.transform.iradon(self.sinogram.T, theta, circle=True, filter=filt_iradon, interpolation='cubic')
        rec = rec.astype(float)[self.n//2+1:-(self.n//2-1),self.n//2:-self.n//2] # adjust iradon shift
        if self.circle:
            astra.extrautils.clipCircle(rec)
        return rec.astype(float)