#!/usr/bin/env python
import argparse, os
import cv2 as cv
from Training import trainDigits, validation, test
from Utils import makeFolder

def main():
  makeFolder("output")
  makeFolder("train")
  makeFolder("test")
  makeFolder("val")
  makeFolder("output/validation")
  
  parser = argparse.ArgumentParser()
  parser.add_argument("-td", '--train_digits', action="store_true", help="this will begin the training process for digits.")
  parser.add_argument("-vd", '--validate_digits', action="store_true", help="this will begin the validation phase.")
  args = parser.parse_args()
  if args.train_digits:
    trainDigits()
  if args.validate_digits:
    try:
      validation()
    except Exception as e:
      print(e)
  test()
  
if __name__ == "__main__":
  main()