import logging
import os
import shutil
import time
from logging import getLogger

import click
import cv2
import enchant
import numpy as np
import pytesseract
from enchant.checker import SpellChecker


@click.command()
@click.option('--input_name',
              default='./samples/ocr_sample.png',
              help='path and name of the input file.')
@click.option('--output_name',
              default='output.txt',
              help='path and name of the output file')
@click.option('--verbose', default=False, help='output detailed logs')
def ocr(input_name, output_name, verbose):
    logging.basicConfig(level=logging.NOTSET)
    logger = logging.getLogger("ocr_test")
    #check input file format
    formats = ['png', 'jpeg', 'pdf']
    if input_name.split('.')[-1] not in formats:
        raise Exception("input format must be png, jpeg, or pdf.")
    if verbose == 'True':
        start_time = time.time()
        logger.info("starting ocr on {}".format(input_name.split('/')[-1]))
    img = cv2.imread(input_name)
    #preprocessing
    img_new = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_new = cv2.threshold(img_new, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    #conduct OCR
    custom_config = '-c -tessedit_write_images=True'
    img_to_str = pytesseract.image_to_string(img_new, config=custom_config)
    #postprocessing: spellcheck
    chkr = enchant.checker.SpellChecker("en_US")
    chkr.set_text(img_to_str)
    for err in chkr:
        sug = err.suggest()[0]
        err.replace(sug)
    img_to_str = chkr.get_text()
    #export to txt file
    with open(output_name, 'w') as fp:
        fp.write(img_to_str)
    if verbose == 'True':
        end_time = time.time()
        logger.info(
            "finishing ocr on {}, saving the txt file as {}. OCR took {} seconds."
            .format(
                input_name.split('/')[-1], output_name,
                round(end_time - start_time, 2)))


if __name__ == '__main__':
    ocr()
