import argparse
from utils.utils import plot_running


def get_args():
    parser = argparse.ArgumentParser(description='Plot result by csv file')
    parser.add_argument('--data_path', '-dp', type=str, default="data/split_data", help='csv file path')

    return parser.parse_args()


def print_debug(message: str):
    print(f"[DEBUG] {message}")


def plot_sample(path: str):
    if ".csv" not in path:
        print_debug("Please set path is csv file")
        return

    plot_running(path)


if __name__ == '__main__':
    args = get_args()
    path = args.data_path
    plot_sample(path)
