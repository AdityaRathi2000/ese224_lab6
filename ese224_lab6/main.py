import numpy as np
import time as time
import sounddevice as sd
from scipy.io.wavfile import write

class dft():
    def __init__(self, x, fs, K=None):
    # START: SANITY CHECK OF INPUTS.
        if (type(fs) != int) or (fs<=0):
            raise NameError('The frequency fs should be a positive integer.')
        if not isinstance(x, np. ndarray):
            raise NameError('The input signal x must be a numpy array.')
        if isinstance(x, np. ndarray):
            if x.ndim!=1:
                raise NameError('The input signal x must be a numpy vector array.')
        self.x=x
        self.fs=fs
        self.N=len(x)
        if K == None:
            K = len(self.x)
        # START: SANITY CHECK OF INPUTS.
        if (type(K) != int) or (K <= 0) or (K < 0):
            raise NameError('K should be a positive integer.')
        self.K=K
        self.f=np.arange(self.K)*self.fs/self.K # (0:K-1) just creates a vector from 0 to K by steps of 1.
        self.f_c=np.arange(-np.ceil(K/2)+1,np.floor(self.K/2)+1)*self.fs/self.K
    def solve3(self):
        X=np.fft.fft(self.x,self.K)/np.sqrt(self.N);
        X_c=np.roll(X,int(np.ceil(self.K/2-1))) # Circularly shift X to get it centered in f_c==0
        return [self.f,X,self.f_c,X_c]


class recordsound():
    def __init__(self, T, fs):
        self.T = T
        self.fs = fs

    def solve(self):
        print('start recording \n')
        voicerecording = sd.rec(int(self.T * self.fs), self.fs, 1)
        sd.wait()  # Wait until recording is finished
        print('end recording \n')
        write('myvoice1.wav', self.fs, voicerecording.astype(np.float32))  # Save as WAV file

        return voicerecording

def q_1_acquire():
    T = 1  # recording time
    fs = 8000  # sampling frequency
    num_recs = 10  # number of recordings for each digit
    digits = [1, 2]  # digits to be recorded
    digit_recs = []

    for digit in digits:
        partial_recs = np.zeros((num_recs, int(T * fs)))
        print('When prompted to speak, say ' + str(digit) + '. \n')
        for i in range(num_recs):
            time.sleep(2)
            digit_recorder = recordsound(T, fs)
            spoken_digit = digit_recorder.solve().reshape(int(T * fs))
            partial_recs[i, :] = spoken_digit
        digit_recs.append(partial_recs)

    # Storing recorded voices
    np.save("recorded_digits.npy", digit_recs)

    # Computing the DFTs
    digit_recs = np.load("recorded_digits.npy")
    digits = [1, 2]
    num_digits = len(digit_recs)
    num_recs, N = digit_recs[0].shape
    fs = 8000
    DFTs = []
    DFTs_c = []

    for digit_rec in digit_recs:
        DFTs_aux = np.zeros((num_recs, N), dtype=np.complex_)
        DFTs_c_aux = np.zeros((num_recs, N), dtype=np.complex_)
        for i in range(num_recs):
            rec_i = digit_rec[i, :]
            # We can use the norm of the ith signal to normalize its DFT
            energy_rec_i = np.linalg.norm(rec_i)
            rec_i /= energy_rec_i
            DFT_rec_i = dft(rec_i, fs)
            [_, X, _, X_c] = DFT_rec_i.solve3()
            DFTs_aux[i, :] = X
            DFTs_c_aux[i, :] = X_c
        DFTs.append(DFTs_aux)
        DFTs_c.append(DFTs_c_aux)

        # Storing DFTs
    np.save("spoken_digits_DFTs.npy", DFTs)
    np.save("spoken_digits_DFTs_c.npy", DFTs_c)

def q_1_testDate():
    T = 1  # recording time
    fs = 8000  # sampling frequency
    num_recs = 10  # number of recordings for the test set
    digit_recs = []

    partial_recs = np.zeros((num_recs, int(T * fs)))
    print('When prompted to speak, say 1 or 2' + '. \n')
    for i in range(num_recs):
        time.sleep(2)
        digit_recorder = recordsound(T, fs)
        spoken_digit = digit_recorder.solve().reshape(int(T * fs))
        partial_recs[i, :] = spoken_digit
    digit_recs.append(partial_recs)

    # Storing recorded voices
    np.save("test_set.npy", partial_recs)

    # Creating an audio file with the spoken digits
    test_set_audio = partial_recs.reshape(T * fs * num_recs)
    file_name = 'test_set_audio_rec.wav'
    write(file_name, fs, test_set_audio.astype(np.float32))


def print_matrix(A, nr_decimals=2):
    # Determine the number of digits in the largest number in the matrix and use
    # it to specify the number format

    nr_digits = np.maximum(np.floor(np.log10(np.amax(np.abs(A)))), 0) + 1
    nr_digits = nr_digits + nr_decimals + 3
    nr_digits = "{0:1.0f}".format(nr_digits)
    number_format = "{0: " + nr_digits + "." + str(nr_decimals) + "f}"

    # Determine matrix size
    n = len(A)
    m = len(A[0])

    # Sweep through rows
    for l in range(m):
        value = " "

        # Sweep through columns
        for k in range(n):
            # ccncatenate entries to create row printout
            value = value + " " + number_format.format(A[k, l])

        # Print row
        print(value)

def q_2_avg_comp():
    T = 1
    fs = 8000
    test_set = np.load("test_set.npy")
    training_set_DFTs = np.abs(np.load("spoken_digits_DFTs.npy"))

    num_digits = len(training_set_DFTs)
    _, N = training_set_DFTs[0].shape
    average_spectra = np.zeros((num_digits, N), dtype=np.complex_)

    for i in range(num_digits):
        average_spectra[i, :] = np.mean(training_set_DFTs[i], axis=0)

    num_recs, N = test_set.shape
    predicted_labels = np.zeros(num_recs)

    for i in range(num_recs):
        rec_i = test_set[i, :]
        energy_rec_i = np.linalg.norm(rec_i)
        rec_i /= energy_rec_i
        DFT_rec_i = dft(rec_i, fs)
        [_, X, _, X_c] = DFT_rec_i.solve3()

        inner_prods = np.zeros(num_digits)
        for j in range(num_digits):
            inner_prods[j] = np.inner(np.abs(X), np.abs(average_spectra[j, :]))
        predicted_labels[i] = np.argmax(inner_prods) + 1

    print("Average spectrum comparison --- predicted labels: \n")
    print_matrix(predicted_labels[:, None], nr_decimals=0)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # run q1
    #q_1_acquire();  #RUN ONCE FOR TRAINING DATA FILE STORAGE
    #q_1_testDate(); #RUN ONCE FOR TEST DATA FILE STORAGE
    q_2_avg_comp()
