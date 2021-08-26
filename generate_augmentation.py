"""
Generate Augmented Data
"""
import argparse
import random

from augmentation import DocumentAugmentation
import os
from shutil import copyfile

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, help="dataset name", required=True)
parser.add_argument("--n", type=int, help="number of sentences", required=True)
parser.add_argument("--augmentation1", type=str, help="augmentation method #1", required=True, default=None)
parser.add_argument("--augmentation2", type=str, help="augmentation method #2", required=True, default=None)
parser.add_argument("--generation_model", type=str, help="generation model for language generation", default='gpt2')
parser.add_argument("--fp16", type=bool, help="flag variable to use fp16", default=False)

args = parser.parse_args()

# read the parameters
DATA_DIR = args.dataset
N = args.n
if args.augmentation1 is not None and args.augmentation2 is not None:
    AUGMENTATION = sorted([args.augmentation1, args.augmentation2])
else:
    print(f"No Valid Augmentation Methods")

fp16 = args.fp16
model = args.generation_model

# Update datasets by performing document augmentation to get augmented dataset
if not os.path.isdir(f"./data/{DATA_DIR}-augmented"):
    # make a new directory for storing the augmented data
    os.mkdir(f"./data/{DATA_DIR}-augmented")

# set the folder name
FOLDER_DIR = AUGMENTATION[0].capitalize() + AUGMENTATION[1].capitalize()

if not os.path.isdir(f"./data/{DATA_DIR}-augmented/{FOLDER_DIR}-NumSent-{N}"):
    os.mkdir(f"./data/{DATA_DIR}-augmented/{FOLDER_DIR}-NumSent-{N}")

    for element in ['train']:
        with open(f"./data/{DATA_DIR}/{element}.source", "r", encoding='utf8') as document:
            for line in document:
                sent = []
                for i in range(len(AUGMENTATION)):
                    method = AUGMENTATION[i]
                    # set the seed
                    if i == 0:
                        random.seed(97)
                    elif i == 1:
                        random.seed(41)
                    augmentation = DocumentAugmentation(n=N, input=line)
                    if method.lower() == 'randominsertion':
                        augmentation.RandomInsertionFromDoc()
                    elif method.lower() == 'randomswap':
                        augmentation.RandomSwap()
                    elif method.lower() == 'randomdelete':
                        augmentation.RandomDeletion()
                    elif method.lower() == 'generation':
                        augmentation.LanguageGenerationReplacement(fp16=fp16, model=model, num_sent_context=N)
                    elif method.lower() == 'rotation':
                        augmentation.DocumentRotation()
                    # record the augmented sentences
                    sent.append(augmentation.augmented_sentences)

                # record - document
                if not os.path.isfile(f"./data/{DATA_DIR}-augmented/{FOLDER_DIR}-NumSent-{N}/{element}.source"):
                    with open(f"./data/{DATA_DIR}-augmented/{FOLDER_DIR}-NumSent-{N}/{element}.source", "w",
                              encoding='utf8') as f:
                        f.write(f"{' '.join(sent[0])}\n")
                        f.write(f"{' '.join(sent[1])}\n")
                else:
                    with open(f"./data/{DATA_DIR}-augmented/{FOLDER_DIR}-NumSent-{N}/{element}.source", "a",
                              encoding='utf8') as f:
                        f.write(f"{' '.join(sent[0])}\n")
                        f.write(f"{' '.join(sent[1])}\n")

        with open(f"./data/{DATA_DIR}/{element}.target", "r", encoding='utf8') as document:
            for line in document:
                # record - summary
                if not os.path.isfile(f"./data/{DATA_DIR}-augmented/{FOLDER_DIR}-NumSent-{N}/{element}.target"):
                    with open(f"./data/{DATA_DIR}-augmented/{FOLDER_DIR}-NumSent-{N}/{element}.target", "w",
                              encoding='utf8') as f:
                        f.write(line)
                        f.write(line)
                else:
                    with open(f"./data/{DATA_DIR}-augmented/{FOLDER_DIR}-NumSent-{N}/{element}.target", "a",
                              encoding='utf8') as f:
                        f.write(line)
                        f.write(line)

    # copy validation
    copyfile(src=f'./data/{DATA_DIR}/val.source',
             dst=f"./data/{DATA_DIR}-augmented/{FOLDER_DIR}-NumSent-{N}/val.source")
    copyfile(src=f'./data/{DATA_DIR}/val.target',
             dst=f"./data/{DATA_DIR}-augmented/{FOLDER_DIR}-NumSent-{N}/val.target")

    # copy test
    copyfile(src=f'./data/{DATA_DIR}/test.source',
             dst=f"./data/{DATA_DIR}-augmented/{FOLDER_DIR}-NumSent-{N}/test.source")
    copyfile(src=f'./data/{DATA_DIR}/test.target',
             dst=f"./data/{DATA_DIR}-augmented/{FOLDER_DIR}-NumSent-{N}/test.target")

else:
    print(f"there is data already in this path: ./data/{DATA_DIR}-augmented/{FOLDER_DIR}-NumSent-{N}")
