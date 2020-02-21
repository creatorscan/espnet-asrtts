#!/bin/bash
set -e
# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh
. ./cmd.sh

dumpdir=dump_80dim   # directory to dump full features
datadir=data_80dim
do_delta=false
fbankdir=fbank_80dim
false && {
utils/combine_data.sh --extra_files utt2num_frames $datadir/train_960_org $datadir/train_clean_100 $datadir/train_clean_360 $datadir/train_other_500
utils/combine_data.sh --extra_files utt2num_frames $datadir/dev_org $datadir/dev_clean $datadir/dev_other

remove_longshortdata.sh --maxframes 3000 --maxchars 400 $datadir/train_960_org $datadir/train_960
remove_longshortdata.sh --maxframes 3000 --maxchars 400 $datadir/dev_org $datadir/dev
for x in train_960 dev_clean test_clean dev_other test_other train_clean_100 train_clean_360 train_other_500; do
echo "stage 1: Feature Generation"
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    if [ ! -s $datadir/${x}/feats.scp ]; then
        bash utils/copy_data_dir.sh data/${x} $datadir/${x}
        steps/make_fbank.sh --cmd "$train_cmd" --nj 32 --write_utt2num_frames true \
            $datadir/${x} exp/make_fbank/${x} ${fbankdir}
    fi

    if [ ! -s $datadir/${x}_org/feats.scp ]; then
    utils/fix_data_dir.sh $datadir/${x}
    utils/copy_data_dir.sh $datadir/${x} $datadir/${x}_org
    fi

    if [ ! -s $datadir/${x}/cmvn.ark ]; then
    # remove utt having more than 3000 frames
    # remove utt having more than 400 characters
    remove_longshortdata.sh --maxframes 3000 --maxchars 400 $datadir/${x}_org $datadir/${x}
    # compute global CMVN
    compute-cmvn-stats scp:$datadir/${x}/feats.scp $datadir/${x}/cmvn.ark
    fi
    if [ ! -s $dumpdir/${x}/feats.scp ]; then
    mkdir -p ${dumpdir}/${x}
    # dump features for training
    dump.sh --cmd "$train_cmd" --nj 30 --do_delta ${do_delta} \
        $datadir/${x}/feats.scp $datadir/${x}/cmvn.ark exp/dump_feats/${x} ${dumpdir}/${x}
    fi
done
}
dict=$datadir/lang_char/train_960_units.txt
nlsyms=$datadir/lang_char/non_lang_syms.txt
echo "dictionary: ${dict}"
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p $datadir/lang_char/
    echo "make a non-linguistic symbol list"
    #cut -f 2- $datadir/train_960/text | tr " " "\n" | sort | uniq | grep "<" > ${nlsyms}
    #cat ${nlsyms}

    echo "make a dictionary"
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    text2token.py -s 1 -n 1 -l ${nlsyms} $datadir/train_960/text | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

    for x in train_960 dev_clean test_clean dev_other test_other train_clean_100 train_clean_360 train_other_500; do
    # make json labels
    data2json.sh --feat ${dumpdir}/${x}/feats.scp --nlsyms ${nlsyms} \
        $datadir/${x} ${dict} > ${dumpdir}/${x}/data.json
done
