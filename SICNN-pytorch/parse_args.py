import argparse

parser = argparse.ArgumentParser(description="SICNN-pytorch Implementation")

parser.add_argument("--train", action="store_true", help="train the model")

args = parser.parse_args()
