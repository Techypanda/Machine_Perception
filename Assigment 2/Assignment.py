#!/usr/bin/env python
import argparse, os
from Training import trainDigits

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("-td", '--train_digits', action="store_true", help="this will begin the training process for digits.")
  args = parser.parse_args()
  if args.train_digits:
    trainDigits()





if __name__ == "__main__":
  main()