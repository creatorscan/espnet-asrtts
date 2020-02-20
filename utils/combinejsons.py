#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import print_function
from __future__ import unicode_literals

import argparse
import codecs
import json
import logging
import os
import sys

from espnet.utils.cli_utils import get_commandline_args

is_python2 = sys.version_info[0] == 2


def get_parser():
    parser = argparse.ArgumentParser(
        description='merge json files',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input-jsons', type=str, nargs='+', action='append',
                        default=[], help='Json files for the inputs')
    parser.add_argument('--verbose', '-V', default=0, type=int,
                        help='Verbose option')
    parser.add_argument('-O', dest='output', type=str, help='Output json file')
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    # logging info
    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    if args.verbose > 0:
        logging.basicConfig(level=logging.INFO, format=logfmt)
    else:
        logging.basicConfig(level=logging.WARN, format=logfmt)
    logging.info(get_commandline_args())

    new_dict = {}  # Dict[str, List[List[Dict[str, Dict[str, dict]]]]]
    # make intersection set for utterance keys
    intersec_ks = None  # Set[str]
    for jsons_list in args.input_jsons:
        for idx, x in enumerate(jsons_list):
            if os.path.isfile(x):
                with codecs.open(x, encoding="utf-8") as f:
                    j = json.load(f)
                ks = list(j['utts'].keys())
                logging.info(x + ': has ' + str(len(ks)) + ' utterances')
                new_dict.update(j['utts'])

    # ensure "ensure_ascii=False", which is a bug
    if args.output is not None:
        sys.stdout = codecs.open(args.output, "w", encoding="utf-8")
    else:
        sys.stdout = codecs.getwriter("utf-8")(
            sys.stdout if is_python2 else sys.stdout.buffer)
    print(json.dumps({'utts': new_dict},
                     indent=4, ensure_ascii=False,
                     sort_keys=True, separators=(',', ': ')))
