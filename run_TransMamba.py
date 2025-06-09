import argparse
from torch.backends import cudnn
from utils.utils import *
from model.TransMamba_Solver import TransMamba_Solver
import time
import warnings
import sys

warnings.filterwarnings('ignore')

# Set the random seed manually for reproducibility.
seed_value = 2024
torch.manual_seed(seed_value)
np.random.seed(seed_value)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_value)

def main(configs):
    cudnn.benchmark = True
    if not os.path.exists(configs.model_save_path):
        mkdir(configs.model_save_path)
    solver = TransMamba_Solver(configs)
    setting = ('_psh{}_psl{}_ws{}_dm{}_hd{}'.format(configs.patch_size_high,
                                                    configs.patch_size_low,
                                                    configs.win_size,
                                                    configs.d_model,
                                                    configs.n_heads))
    if configs.mode == 'train':
        solver.train(setting)
    elif configs.mode == 'test':
        solver.test(setting)
    return solver

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # OPTIMIZATION
    parser.add_argument('--num_epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--dropout', type=float, default=.1, help='dropout')
    parser.add_argument('--lr', type=float, default=1e-4, help='optimizer learning rate')
    parser.add_argument('--loss_fuc', type=str, default='MSE', help='loss function', choices=['MAE', 'MSE'])
    parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')

    # Model Parameter
    parser.add_argument('--n_heads', type=int, default=1, help='multi head attention')  # Transformer
    parser.add_argument('--e_layers', type=int, default=1, help='num of encoder layers')  # Model

    # Anomaly Detection Task
    parser.add_argument('--show_params', type=int, default=1)
    parser.add_argument('--aug_rate', type=float, default=0.1, help='Augmentation rates: [point, masking, segment]')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
    parser.add_argument('--model_save_path', type=str, default='checkpoints')
    parser.add_argument('--patch_size_high', type=int, nargs='+', default=[1, 3])
    parser.add_argument('--patch_size_low', type=int, nargs='+', default=[2, 6])
    parser.add_argument('--d_model', type=int, default=64, help='Embedded representation')
    parser.add_argument('--dataset', type=str, default='PSM')
    parser.add_argument('--channel', type=int, default=25)
    parser.add_argument('--win_size', type=int, default=90, help='sliding window size')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--anomaly_ratio', type=float, default=1)

    config = parser.parse_args()

    # Results
    if not os.path.exists("result"):
        mkdir("result")
    sys.stdout = Logger("result/" + config.dataset + ".log", sys.stdout)

    args = vars(config)
    if config.mode == 'train':
        print("\n")
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        print('================ Hyperparameters ===============')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print(f"cuda_num = {torch.cuda.device_count()}")
    else:
        print("\n")
        print('================ Test: Hyperparameters ===============')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))

    main(config)
