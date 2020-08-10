import numpy as np
import matplotlib
from matplotlib import animation
from scipy.signal import lfilter, lfilter_zi, firwin
from time import sleep
from pylsl import StreamInlet, resolve_byprop
import seaborn as sns
from scipy.signal import spectrogram

VIEW_BUFFER = 12
VIEW_SUBSAMPLE = 2
LSL_SCAN_TIMEOUT = 5
LSL_EEG_CHUNK = 12



class LSLViewer():
    def __init__(self, stream, fig, axes, window, scale, dejitter=True):
        """Init"""
        self.stream = stream
        self.window = window
        self.scale = scale
        self.dejitter = dejitter
        self.inlet = StreamInlet(stream, max_chunklen=LSL_EEG_CHUNK)
        self.filt = True
        self.subsample = VIEW_SUBSAMPLE

        info = self.inlet.info()
        description = info.desc()

        self.sfreq = info.nominal_srate()
        self.n_samples = int(self.sfreq * self.window)
        self.n_chan = info.channel_count()

        ch = description.child('channels').first_child()
        ch_names = [ch.child_value('label')]

        for i in range(self.n_chan):
            ch = ch.next_sibling()
            ch_names.append(ch.child_value('label'))

        self.ch_names = ch_names

        fig.canvas.mpl_connect('key_press_event', self.OnKeypress)
        fig.canvas.mpl_connect('button_press_event', self.onclick)

        self.fig = fig
        self.axes = axes

        sns.despine(left=True)

        self.data = np.zeros((self.n_samples, self.n_chan))
        self.times = np.arange(-self.window, 0, 1. / self.sfreq)
        impedances = np.std(self.data, axis=0)

        f, t, S = self.getspec(self.data[:, 0])
        self.im = self.axes.imshow(S, cmap=matplotlib.pyplot.cm.Reds, vmin=0, vmax=1,
                                   extent=[t[0], t[-1], f[0], f[-1]])

        axes.set_xlabel('Time (s)')
        axes.xaxis.grid(False)

        axes.set_ylabel('Frequency (Hz)')

        self.display_every = int(0.2 / (12 / self.sfreq))
        self.bf = firwin(32, np.array([1, 40]) / (self.sfreq / 2.), width=0.05,
                         pass_zero=False)
        self.af = [1.0]

        zi = lfilter_zi(self.bf, self.af)
        self.filt_state = np.tile(zi, (self.n_chan, 1)).transpose()
        self.data_f = np.zeros((self.n_samples, self.n_chan))

    def update_plot(self, k):
        try:
            samples, timestamps = self.inlet.pull_chunk(timeout=1.0,
                                                        max_samples=LSL_EEG_CHUNK)
            if timestamps:
                if self.dejitter:
                    timestamps = np.float64(np.arange(len(timestamps)))
                    timestamps /= self.sfreq
                    timestamps += self.times[-1] + 1. / self.sfreq
                self.times = np.concatenate([self.times, timestamps])
                self.n_samples = int(self.sfreq * self.window)
                self.times = self.times[-self.n_samples:]
                self.data = np.vstack([self.data, samples])
                self.data = self.data[-self.n_samples:]
                filt_samples, self.filt_state = lfilter(
                    self.bf, self.af,
                    samples,
                    axis=0, zi=self.filt_state)
                self.data_f = np.vstack([self.data_f, filt_samples])
                self.data_f = self.data_f[-self.n_samples:]
                k += 1
                if k % self.display_every == 0:
                    if self.filt:
                        plot_data = self.data_f
                    elif not self.filt:
                        plot_data = self.data - self.data.mean(axis=0)

                    _, _, S = self.getspec(plot_data[:, 0])
                    self.im.set_data(S)
            else:
                sleep(0.2)
        except RuntimeError as e:
            raise

    def getspec(self, data):
        return spectrogram(data, fs=1/256.0, nperseg=64, noverlap=48)

    def onclick(self, event):
        print((event.button, event.x, event.y, event.xdata, event.ydata))

    def OnKeypress(self, event):
        if event.key == '/':
            self.scale *= 1.2
        elif event.key == '*':
            self.scale /= 1.2
        elif event.key == '+':
            self.window += 1
        elif event.key == '-':
            if self.window > 1:
                self.window -= 1
        elif event.key == 'd':
            self.filt = not(self.filt)


def view(window, scale, refresh, figure, backend, version=1):
    matplotlib.use(backend)
    sns.set(style="whitegrid")

    figsize = np.int16(figure.split('x'))

    print("Looking for an EEG stream...")
    streams = resolve_byprop('type', 'EEG', timeout=LSL_SCAN_TIMEOUT)

    if len(streams) == 0:
        raise(RuntimeError("Can't find EEG stream."))
    print("Start acquiring data.")

    fig, axes = matplotlib.pyplot.subplots(1, 1, figsize=figsize, sharex=True)
    lslv = LSLViewer(streams[0], fig, axes, window, scale)
    #fig.canvas.mpl_connect('close_event', lslv.stop)

    help_str = """
                toggle filter : d
                toogle full screen : f
                zoom out : /
                zoom in : *
                increase time scale : -
                decrease time scale : +
               """
    print(help_str)
    anim = animation.FuncAnimation(fig, lslv.update_plot, interval=50)
    matplotlib.pyplot.show()

if __name__ == "__main__":
    view(5, 100, 0.2, "15x6", "Qt5Agg")
