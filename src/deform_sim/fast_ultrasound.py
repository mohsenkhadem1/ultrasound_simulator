"""Coherent beamformed-IQ PSF preview, intentionally distinct from RF/SIMUS."""
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates


class FastUltrasound:
    def __init__(self, size=192, sound_speed=1540., frequency=5e6,
                 lateral_min=-.02, lateral_max=.02, depth_min=.002, depth_max=.05):
        self.size, self.sound_speed, self.frequency = size, sound_speed, frequency
        self.calibration = dict(lateral_min=lateral_min, lateral_max=lateral_max,
                                depth_min=depth_min, depth_max=depth_max)
        self.reference = None
        self.x = np.linspace(lateral_min, lateral_max, 256)
        self.z = np.linspace(depth_min, depth_max, max(768, int(np.ceil((depth_max-depth_min)/.000065))))
        self.sampling = np.meshgrid(np.linspace(0, len(self.z)-1, size), np.linspace(0, 255, size), indexing='ij')

    def render(self, xyz, coefficients):
        keep = (xyz[:, 2]>=self.z[0])&(xyz[:, 2]<=self.z[-1])&(xyz[:, 0]>=self.x[0])&(xyz[:, 0]<=self.x[-1])&(np.abs(xyz[:, 1])<.008)
        points, amplitudes = xyz[keep], coefficients[keep]
        grid = np.zeros((len(self.z), len(self.x)), np.complex64)
        if len(points):
            row = (points[:, 2]-self.z[0])/(self.z[-1]-self.z[0])*(len(self.z)-1)
            col = (points[:, 0]-self.x[0])/(self.x[-1]-self.x[0])*(len(self.x)-1)
            r, c = np.floor(row).astype(int), np.floor(col).astype(int)
            elevation_sigma = .0012+.035*np.abs(points[:, 2]-.02)
            elevation = np.exp(-.5*(points[:, 1]/elevation_sigma)**2)
            phase = 4*np.pi*self.frequency*points[:, 2]/self.sound_speed
            amplitude = amplitudes*elevation*np.exp(1j*phase)
            for dr, dc in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                rr, cc = r+dr, c+dc
                weight = (row-r if dr else 1-row+r)*(col-c if dc else 1-col+c)
                valid = (rr>=0)&(rr<len(self.z))&(cc>=0)&(cc<len(self.x))
                np.add.at(grid, (rr[valid], cc[valid]), (amplitude[valid]*weight[valid]).astype(np.complex64))
        # Complex summation precedes envelope detection: coherent speckle,
        # finite axial/lateral resolution and an elevation sensitivity profile.
        iq = gaussian_filter(grid, [.00012/(self.z[1]-self.z[0]), .00045/(self.x[1]-self.x[0])], mode='constant')
        envelope = np.sqrt(map_coordinates((iq.real**2+iq.imag**2), self.sampling, order=1, mode='constant'))
        x, z = np.meshgrid(np.linspace(self.x[0], self.x[-1], self.size), np.linspace(self.z[0], self.z[-1], self.size))
        aperture = 1/(1+np.exp(np.clip((np.abs(x)-(.0096+.28*z))/.0006, -40, 40)))
        envelope *= aperture
        if self.reference is None and envelope.max()>1e-7:
            self.reference = max(float(np.percentile(envelope, 99.95))*2, float(envelope.max())*1.05, 1e-6)
        image = np.clip((20*np.log10(np.maximum(envelope, 1e-30)/max(self.reference or 1., 1e-20))+60)/60, 0, 1)
        return image.astype(np.float32)
