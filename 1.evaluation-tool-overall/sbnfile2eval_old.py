import logging
from argparse import ArgumentParser, Namespace
import concurrent.futures
import tempfile
from typing import Any, Dict, Generator, Optional
from tqdm.contrib.logging import logging_redirect_tqdm
from tqdm import tqdm
import pandas as pd
from ud_boxer.sbn import SBNGraph, SBNSource
from datetime import datetime
import os
from ud_boxer.helpers import create_record, smatch_score
from ud_boxer.sbn_spec import SBNError, get_base_id, get_doc_id
from ud_boxer.misc import ensure_ext
from os import PathLike
import json
import subprocess
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "-gold",
        "--gold_sbn",
        type=str,
        required=True,
        help="Path to sequential gold SBN file, the format should be path | raw | sbn",
    )
    parser.add_argument(
        "-test",
        "--test_sbn",
        type=str,
        required=True,
        help="Path to seq2seq output file.",
    )
    parser.add_argument(
        "-lang",
        "--lang",
        type=str,
        default="en",
        help="the language to evaluate.",
    )

    parser.add_argument(
        "--sbn_source",
        default=SBNSource.SEQ2SEQ.value,
        type=str,
        choices=SBNSource.all_values(),
        help="Add flag to SBNGraph and results where this file came from.",
    )
    parser.add_argument(
        "-r",
        "--results_path",
        type=str,
        default="./results",
        help="CSV file to write results and scores to.",
    )
    parser.add_argument(
        "-w",
        "--max_workers",
        default=16,
        help="Max concurrent workers used to run inference with. Be careful "
             "with setting this too high since mtool might error (segfault) if hit "
             "too hard by too many concurrent tasks.",
    )
    return parser.parse_args()


_KEY_MAPPING = {
    "n": "input_graphs",
    "g": "gold_graphs_generated",
    "s": "evaluation_graphs_generated",
    "c": "correct_graphs",
    "p": "precision",
    "r": "recall",
    "f": "f1",
}
_RELEVANT_ITEMS = ["p", "r", "f"]


def smatch_score(gold: PathLike, test: PathLike, test_penman_file, gold_penman_file, store_penman: bool = True) -> Dict[str, float]:
    """Use mtool to score two amr-like graphs using SMATCH"""
    try:
        # NOTE: this is not ideal, but mtool is quite esoteric in how it reads
        # in graphs, so it's quite hard to just plug two amr-like strings
        # in it. Maybe we can run this as a deamon to speed it up a bit or
        # put some time into creating a usable package to import for this use-
        # case.
        smatch_cmd = f"mtool --read amr --score smatch --gold {gold} {test}"
        response = subprocess.check_output(smatch_cmd, shell=True)
        decoded = json.loads(response)
        if store_penman:
            with open(test_penman_file, 'a') as f1, open(gold_penman_file, 'a') as f2, \
                    open(test, 'r') as test_file, open(gold, 'r') as gold_file:  # f1 test, f2 gold
                f1.writelines(test_file.readlines())
                f1.write('\n' + '\n')
                f2.writelines(gold_file.readlines())
                f2.write('\n' + '\n')

    except subprocess.CalledProcessError as e:
        raise SBNError(
            f"Could not call mtool smatch with command '{smatch_cmd}'\n{e}"
        )
    clean_dict = {
        _KEY_MAPPING.get(k, k): v
        for k, v in decoded.items()
        if k in _RELEVANT_ITEMS
    }
    return clean_dict

def  write_penman_file(gold_file, test_file, test_penman_file, gold_penman_file):
    with open(test_penman_file, 'a') as f1, open(gold_penman_file, 'a') as f2,\
            open(test_file, 'r') as test, open(gold_file, 'r') as gold:  # f1 test, f2 gold
        f1.writelines(test.readlines())
        f1.write('\n' + '\n')
        f2.writelines(gold.readlines())
        f2.write('\n' + '\n')

def generate_result(args, test_sbn_line, gold_sbn_line, test_penman_file, gold_penman_file):
    print(test_sbn_line)
    G1 = SBNGraph(source=args.sbn_source).from_string(gold_sbn_line)
    G2 = SBNGraph(source=args.sbn_source).from_string(test_sbn_line)
    lenient_err, strict_err = None, None
    with tempfile.NamedTemporaryFile("w") as f1, tempfile.NamedTemporaryFile("w") as f2:
        try:
            strict_scores = smatch_score(G1.to_penman(f1.name), G2.to_penman(f2.name), test_penman_file, gold_penman_file)
            # write_penman_file(G1.to_penman(f1.name), G2.to_penman(f2.name), test_penman_file, gold_penman_file) ## get the penman format files ## strict
        except SBNError as e_strict:
            strict_scores = dict()
            strict_err = str(e_strict)
        try:
            lenient_scores = smatch_score(G1.to_penman(f1.name), G2.to_penman(f2.name, strict=False), test_penman_file, gold_penman_file, store_penman=False)
        except SBNError as e:
            lenient_scores = dict()
            lenient_err = str(e)
    return (
        strict_scores,
        lenient_scores,
        G2.to_sbn_string(),
        lenient_err,
        strict_err,
    )

def full_run(args, test_sbn_line, gold_sbn, test_penman_file, gold_penman_file):
    path = gold_sbn.split("\t")[0]
    raw_sent = gold_sbn.split("\t")[1]
    if args.lang == "en":
        gold_sbn_line = gold_sbn.split("\t")[-3]
    if args.lang =="zh":
        gold_sbn_line = gold_sbn.split("\t")[-2]
    if args.lang =="order":
        gold_sbn_line = gold_sbn.split("\t")[-1]
    sbn, lenient_error, strict_error = None, None, None
    strict_scores, lenient_scores = dict(), dict()
    try:
        (
            strict_scores,
            lenient_scores,
            sbn,
            lenient_error,
            strict_error,
        ) = generate_result(args, test_sbn_line, gold_sbn_line, test_penman_file, gold_penman_file)
    except Exception as e:
        logger.error(e)

    record = create_record(
        pmb_id=path,
        raw_sent=raw_sent,
        sbn_source=args.sbn_source,
        sbn=sbn,
        lenient_error=lenient_error,
        strict_error=strict_error,
        strict_scores=strict_scores,
        lenient_scores=lenient_scores,
    )
    return record

def main():
    args = get_args()
    if args.results_path:
        test_penman_file = os.path.join(args.results_path, "test.penman")
        gold_penman_file = os.path.join(args.results_path, "gold.penman")
    with open(args.test_sbn, 'r') as test_input, open(args.gold_sbn, 'r') as gold_input:
        test_sbn_list = []
        gold_sbn_list = []
        for line in test_input:
            test_sbn_list.append(line.strip())
        for line in gold_input:
            gold_sbn_list.append(line.strip())
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.max_workers
    ) as executor:
        futures = []
        for index, gold_sbn_line in enumerate(gold_sbn_list):
            test_sbn_line = test_sbn_list[index]
            futures.append(executor.submit(full_run, args, test_sbn_line, gold_sbn_line, test_penman_file, gold_penman_file))
        result_records = [
            res.result()
            for res in tqdm(
                concurrent.futures.as_completed(futures),
                desc="Running inference",
            )
        ]
    df = pd.DataFrame().from_records(result_records)
    if args.results_path:
        final_path = os.path.join(args.results_path, "smatch.scores")
        df.to_csv(final_path, index=False)

    df["f1"] = df["f1"].fillna(0)
    df["f1_lenient"] = df["f1_lenient"].fillna(0)
    generation_data = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    overall_result_msg = f"""
        {generation_data}

        ARGS: {args}

        PARSED DOCS:          {len(df[df['lenient_error'].isnull()])}
        FAILED DOCS:          {len(df[df['lenient_error'].notnull()])}
        TOTAL DOCS:           {len(df)}

        AVERAGE F1 (strict):  {df["f1"].mean():.3} ({df["f1"].min():.3} - {df["f1"].max():.3})
        AVERAGE F1 (lenient): {df["f1_lenient"].mean():.3} ({df["f1_lenient"].min():.3} - {df["f1_lenient"].max():.3})
        """
    overall_result = os.path.join(args.results_path, "overall.txt")
    with open(overall_result, "a") as f:
        f.write(f"{overall_result_msg}\n\n")

    print(overall_result_msg)


if __name__ == "__main__":
    with logging_redirect_tqdm():
        main()
