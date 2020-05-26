from fsite.ournet import OurNet
from utils.utils import default_env
from data.synthetic_data import get_ihdp_npci
import sys
import time


class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open(f'results/{time.strftime("%Y%m%d-%H%M%S")}.txt', 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


sys.stdout = Logger()
default_env(gpu=True)
for i, row in enumerate(get_ihdp_npci()):
    print(f'{i+1}/1000')
    X_train, T_train, Yf_train, Ycf_train = row['train']
    X_val, T_val, Yf_val, Ycf_val = row['val']
    X_test, T_test, Yf_test, Ycf_test = row['test']
    n_features = [0, X_train.shape[1], 0]
    n_treatments = T_train.max() + 1

    drcfr = OurNet(n_features)
    drcfr.train(X_train, T_train, Yf_train, 3000, Ycf=Ycf_train,
                val_data=(X_val, T_val, Yf_val, Ycf_val),
                test_data=(X_test, T_test, Yf_test, Ycf_test),
                verbose=True)
    print()
