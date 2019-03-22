import argparse

parser = argparse.ArgumentParser(description="SICNN-pytorch Implementation")

parser.add_argument("--train", action="store_true", help="train the model")

parser.add_argument("--epochs", action="store", type=int, help="epochs to train")
parser.add_argument("--bs", action="store", type=int, help="batch size")

args = parser.parse_args()
