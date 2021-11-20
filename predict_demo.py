import os
import argparse

parser = argparse.ArgumentParser(
    epilog="Example: python dataset.py"
)
parser.add_argument(
    "--input-dir",
    help="The parameter to add to the calculation",
    dest="input_dir",
    type=str,
    required=True
)
args = parser.parse_args()

directory = args.input_dir
hashes = set([fp.split('_')[2] for fp in os.listdir(directory)])
for hash in hashes:
    command = f'python predict_main.py --test-filepath {directory}/test_data_{hash}'
    os.system(command)
