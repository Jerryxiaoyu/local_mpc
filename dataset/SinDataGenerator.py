import numpy as np
import matplotlib.pyplot as plt

CONFIG = {
    'amplitude_range': [0.1, 5.0],
    'phase_range': [0, np.pi],
    'x_range': [-5.0, 5.0],
    'visualize_sample_point': 100
}


class dataset(object):
    def __init__(self, K_shots=5):
        # K for meta train, and another K for meta val
        self._K = K_shots*2
        self.dim_input = 1
        self.dim_output = 1
        self.name = 'sin'

    def _resample_task(self, batch_size, verbose):
        self._sample_amplitude = np.random.uniform(CONFIG['amplitude_range'][0],
                                                   CONFIG['amplitude_range'][1], batch_size)
        self._sample_phase = np.random.uniform(CONFIG['phase_range'][0],
                                               CONFIG['phase_range'][1], batch_size)
        # Match the shape of input
        self._sample_amplitude_tile = np.tile(self._sample_amplitude, [self._K, self.dim_input, 1])
        self._sample_amplitude_tile = np.transpose(self._sample_amplitude_tile, [2, 0, 1])
        self._sample_phase_tile = np.tile(self._sample_phase, [self._K, self.dim_input, 1])
        self._sample_phase_tile = np.transpose(self._sample_phase_tile, [2, 0, 1])

        
        if verbose:
            print("Mean of the amplitude", np.mean(self._sample_amplitude))
            print("Mean of the phase", np.mean(self._sample_phase))

    def get_batch(self, batch_size, resample=False, verbose=False):
        if resample:
            self._resample_task(batch_size, verbose)
        x = np.random.uniform(CONFIG['x_range'][0], CONFIG['x_range'][1],
                              [batch_size, self._K, self.dim_input])
        y = self._sample_amplitude_tile*np.sin(x - self._sample_phase_tile)
        return x, y, self._sample_amplitude, self._sample_phase
    
    def get_libatch(self,batch_size, resample=False, verbose=False):
        if resample:
            self._resample_task(batch_size, verbose)
        x_point = np.random.uniform(CONFIG['x_range'][0], CONFIG['x_range'][1],
                              [batch_size, self._K, self.dim_input])
        y_point = self._sample_amplitude_tile * np.sin(x_point - self._sample_phase_tile)
        x = np.linspace(CONFIG['x_range'][0], CONFIG['x_range'][1], CONFIG['visualize_sample_point'])
        x = x[np.newaxis, :, np.newaxis]
        x = np.tile(x, (1, 1, 1))
        sample_amplitude_tile =np.tile(self._sample_amplitude_tile,(1,10,1))
        sample_phase_tile =np.tile(self._sample_phase_tile,(1,10,1))
        y =   sample_amplitude_tile * np.sin(x -   sample_phase_tile)
        return x_point,y_point,x, y, self._sample_amplitude, self._sample_phase
        
        
    def visualize(self, amplitude, phase, inp, output, path):
        inp = np.squeeze(inp)
        output = np.squeeze(output)
        x = np.linspace(CONFIG['x_range'][0], CONFIG['x_range'][1], CONFIG['visualize_sample_point'])
        curve = amplitude*np.sin(x - phase)
        # GT curve
        plt.plot(x, curve, '.')
        # prediction
        plt.plot(inp, output, 'v')
        plt.savefig(path)
        plt.close()


    def visualize2(self, amplitude, phase, inp, output, path):
        inp = np.squeeze(inp)
        output = np.squeeze(output)
        x = np.linspace(CONFIG['x_range'][0], CONFIG['x_range'][1], CONFIG['visualize_sample_point'])
        curve = amplitude * np.sin(x - phase)
        # GT curve
        plt.plot(x, curve, )
        # target
        plt.plot(inp, amplitude * np.sin(inp - phase), 'v')

        
        plt.plot(x, output, '.',color='g')
        plt.savefig(path)
        plt.close()