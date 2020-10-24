#!/usr/bin/env python
import argparse
import os
import cv2 as cv
from Training import trainDigits, validation, test
from Utils import makeFolder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-td", '--train_digits',
                        help="This will train digits based off data in directory (NOTE: it expects a certain format as its hardcoded labels.).")
    parser.add_argument("-vd", '--validate_digits',
                        help="This will begin the validation phase in given directory.")
    parser.add_argument(
        "-t", '--test', help="This will begin testing in given directory.", type=str)
    parser.add_argument(
        "-d", '--debug', help="Enable Debug Statements", default=False, type=bool)
    parser.add_argument("-o", "--output", default="output/")
    args = parser.parse_args()

    if args.output == "output/":
        makeFolder("output")
    makeFolder("{}/validation".format(args.output))

    if args.train_digits:
        trainDigits(args.train_digits)
    if args.validate_digits:
        try:
            validation(args.validate_digits, args.output)
        except Exception as e:
            print(e)
    if args.test:
        test(args.test, args.output, args.debug)


if __name__ == "__main__":
    main()
