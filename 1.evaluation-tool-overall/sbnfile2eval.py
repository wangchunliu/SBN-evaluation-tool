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
from ud_boxer.sbn_spec import SBNError, get_base_id, get_doc_id
from ud_boxer.misc import ensure_ext
from os import PathLike
import json
import subprocess
import re
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

def create_record(
    pmb_id: str,
    raw_sent: str,
    sbn_source: SBNSource = SBNSource.UNKNOWN,
    strict_scores: Dict[str, Any] = dict(),
    lenient_scores: Dict[str, Any] = dict(),
    sbn: Optional[str] = None,
    lenient_error: Optional[str] = None,
    strict_error: Optional[str] = None,
):
    return {
        "pmb_id": pmb_id,
        "source": sbn_source,
        "raw_sent": raw_sent,
        "sbn_str": sbn,
        "lenient_error": lenient_error,
        "strict_error": strict_error,
        **strict_scores,
        **{f"{k}_lenient": v for k, v in lenient_scores.items()},
    }


def smatch_score(gold: PathLike, test: PathLike) -> Dict[str, float]:
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

def generate_result(args, test_sbn_line, path):
    gold_strict_path = os.path.join(args.results_path, path, f"{args.lang}.gold.strict.penman")
    gold_lenient_path = os.path.join(args.results_path, path, f"{args.lang}.gold.lenient.penman")
    test_path_strict = os.path.join(args.results_path, path, f"{args.lang}.test.strict")
    test_path_lenient = os.path.join(args.results_path, path, f"{args.lang}.test.lenient")
    G = SBNGraph(source=args.sbn_source).from_string(test_sbn_line)
    lenient_err, strict_err = None, None
    with open(test_path_strict, 'w') as f_s, open(test_path_lenient, 'w') as f_l:
        try:
            strict_scores = smatch_score(gold_strict_path, G.to_penman(f_s.name))
        except SBNError as e_strict:
            strict_scores = dict()
            strict_err = str(e_strict)
        try:
            lenient_scores = smatch_score(gold_lenient_path, G.to_penman(f_l.name, strict=False))
        except SBNError as e:
            lenient_scores = dict()
            lenient_err = str(e)
    return (
        strict_scores,
        lenient_scores,
        G.to_sbn_string(),
        lenient_err,
        strict_err,
    )


def full_run(args, test_sbn_line, gold_sbn):
    path = gold_sbn.split("\t")[0]
    raw_sent = gold_sbn.split("\t")[1]
    sbn, lenient_error, strict_error = None, None, None
    strict_scores, lenient_scores = dict(), dict()
    try:
        (
            strict_scores,
            lenient_scores,
            sbn,
            lenient_error,
            strict_error,
        ) = generate_result(args, test_sbn_line, path)
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

def store_penman(args, gold_sbn):
    for file_line in gold_sbn:
        path = file_line.split("\t")[0]
        filepath = os.path.join(args.results_path, path)
        isExists = os.path.exists(filepath)
        if not isExists:
            os.makedirs(filepath)
    for file_line in gold_sbn:
        path = file_line.split("\t")[0]
        filepath_strict = os.path.join(args.results_path, path, f"{args.lang}.gold.strict.penman")
        filepath_lenient = os.path.join(args.results_path, path, f"{args.lang}.gold.lenient.penman")
        gold_sbn_line = file_line.split("\t")[-1]
        try:
            G = SBNGraph().from_string(gold_sbn_line)
            G.to_penman(filepath_strict)
            G.to_penman(filepath_lenient, strict=False)
        except SBNError as e:
            logger.warning(e)

def merge_penman(args, gold_sbn_list):
    one_test_penman = os.path.join(args.results_path, f"{args.lang}.test.penman")
    one_gold_penman = os.path.join(args.results_path, f"{args.lang}.gold.penman")
    with open(one_test_penman, 'w') as testfile, open(one_gold_penman, 'w') as goldfile:
        for file_line in gold_sbn_list:
            path = file_line.split("\t")[0]
            test_filepath = os.path.join(args.results_path, path, f"{args.lang}.test.strict.penman")
            gold_filepath = os.path.join(args.results_path, path, f"{args.lang}.gold.strict.penman")
            if os.path.exists(test_filepath) and os.path.getsize(test_filepath):
                f1 = open(test_filepath, 'r')
                testfile.writelines(f1.readlines())
                testfile.write('\n' + '\n')
                f1.close()
                f2 = open(gold_filepath, 'r')
                goldfile.writelines(f2.readlines())
                goldfile.write('\n' + '\n')
                f2.close()


def main():
    args = get_args()
    with open(args.test_sbn, 'r') as test_input, open(args.gold_sbn, 'r') as gold_input:
        test_sbn_list = []
        gold_sbn_list = []
        for line in test_input:
            test_sbn_list.append(line.strip())
        for line in gold_input:
            gold_sbn_list.append(line.strip())
    store_penman(args, gold_sbn_list)
    print("Finish Gold data penman converting!")
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.max_workers
    ) as executor:
        futures = []
        for index, gold_sbn_line in enumerate(gold_sbn_list):
            test_sbn_line = test_sbn_list[index]
            futures.append(executor.submit(full_run, args, test_sbn_line, gold_sbn_line))
        result_records = [
            res.result()
            for res in tqdm(
                concurrent.futures.as_completed(futures),
                desc="Running inference",
            )
        ]
    merge_penman(args, gold_sbn_list)
    print("Merging penman files to one!")
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
