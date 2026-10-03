"""Stream licensed JSONL corpus shards into private token files, never weights."""

from __future__ import annotations

import argparse
from array import array
import hashlib
from pathlib import Path
import shutil
import sqlite3
import sys

from training.common import (
    PipelineError, ROOT, SPLITS, architecture_lock, iter_records, normalize,
    private_output, private_root, prompt_text, sha256_file, verify_tokenizer_vocabulary, write_json,
)


def fingerprints(text: str) -> list[str]:
    words = normalize(text).split()
    grams = {" ".join(words[index:index + 3]) for index in range(max(0, len(words) - 2))}
    return sorted({hashlib.sha256(gram.encode()).hexdigest()[:16] for gram in grams})[:16]


class LeakageIndex:
    """Disk-backed exact checks and a conservative sampled shingle check."""

    def __init__(self, path: Path):
        self.connection = sqlite3.connect(path)
        self.connection.executescript("""
            CREATE TABLE records (id TEXT PRIMARY KEY, digest TEXT UNIQUE, split TEXT, answer_digest TEXT);
            CREATE INDEX answer_lookup ON records(answer_digest, split);
            CREATE TABLE groups (id TEXT PRIMARY KEY, split TEXT);
            CREATE TABLE fingerprints (value TEXT, record_id TEXT, split TEXT);
            CREATE INDEX fingerprint_lookup ON fingerprints(value, split);
        """)

    def add(self, record: dict) -> None:
        split = record["split"]
        previous = self.connection.execute("SELECT split FROM groups WHERE id=?", (record["group_id"],)).fetchone()
        if previous and previous[0] != split:
            raise PipelineError("A source/topic group crosses splits.")
        self.connection.execute("INSERT OR IGNORE INTO groups VALUES (?,?)", (record["group_id"], split))
        digest = hashlib.sha256(normalize(prompt_text(record)).encode()).hexdigest()
        answer_digest = None
        if "messages" in record:
            answer = normalize(record["messages"][-1]["content"])
            if len(answer.split()) >= 8:
                answer_digest = hashlib.sha256(answer.encode()).hexdigest()
                if self.connection.execute("SELECT 1 FROM records WHERE answer_digest=? AND split!=? LIMIT 1", (answer_digest, split)).fetchone():
                    raise PipelineError("A duplicate answer leaks across splits.")
        marks = fingerprints(prompt_text(record))
        if len(marks) >= 12:
            placeholders = ",".join("?" for _ in marks)
            query = f"SELECT record_id, COUNT(*) FROM fingerprints WHERE split!=? AND value IN ({placeholders}) GROUP BY record_id HAVING COUNT(*)>=12 LIMIT 1"
            if self.connection.execute(query, (split, *marks)).fetchone():
                raise PipelineError("A likely near-duplicate crosses splits; review the corpus.")
        try:
            self.connection.execute("INSERT INTO records VALUES (?,?,?,?)", (record["id"], digest, split, answer_digest))
        except sqlite3.IntegrityError as error:
            raise PipelineError("Duplicate dataset ID or normalized prompt/document.") from error
        self.connection.executemany("INSERT INTO fingerprints VALUES (?,?,?)", [(mark, record["id"], split) for mark in marks])

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, nargs="+", default=[ROOT / "data" / "starter.jsonl"])
    parser.add_argument("--output", type=Path, default=private_root() / "corpus")
    parser.add_argument("--download-tokenizer", action="store_true", help="Explicitly download the pinned tokenizer files only; no language-model weights.")
    args = parser.parse_args()
    output = private_output(args.output)
    if output.exists():
        parser.exit(1, "Use a new private corpus output directory.\n")
    lock = architecture_lock()
    # Tokenizer loading is the only network-capable call in corpus preparation.
    try:
        from transformers import AutoTokenizer
    except ImportError:
        parser.exit(1, "Install training/requirements.txt in an isolated environment first.\n")
    cache = private_output(private_root() / "cache")
    tokenizer = AutoTokenizer.from_pretrained(
        lock["tokenizer"]["model_id"], revision=lock["tokenizer"]["revision"],
        cache_dir=str(cache), local_files_only=not args.download_tokenizer,
        trust_remote_code=False,
    )
    verify_tokenizer_vocabulary(len(tokenizer))
    output.mkdir(parents=True, mode=0o700)
    index = LeakageIndex(output / "leakage-index.sqlite")
    counts = {split: 0 for split in SPLITS}
    token_counts = {split: 0 for split in SPLITS}
    unreviewed = synthetic = 0
    handles = {split: (output / f"{split}.tokens").open("wb") for split in SPLITS}
    try:
        for record in iter_records(args.data):
            index.add(record)
            if "text" in record:
                ids = tokenizer.encode(record["text"], add_special_tokens=False) + [tokenizer.eos_token_id]
            else:
                ids = tokenizer.apply_chat_template(record["messages"], tokenize=True, add_generation_prompt=False)
            if any(type(token) is not int or not 0 <= token < len(tokenizer) for token in ids):
                raise PipelineError("Tokenizer emitted an invalid token ID.")
            tokens = array("I", ids)
            if tokens.itemsize != 4:
                raise PipelineError("Token files require four-byte unsigned integers.")
            if sys.byteorder != "little":
                tokens.byteswap()
            handles[record["split"]].write(tokens.tobytes())
            counts[record["split"]] += 1
            token_counts[record["split"]] += len(ids)
            unreviewed += not record["provenance"]["reviewed"]
            synthetic += record["provenance"]["kind"] == "authored_synthetic"
            if sum(counts.values()) % 10_000 == 0:
                index.connection.commit()
        if any(not count for count in counts.values()):
            raise PipelineError("Corpus requires nonempty train, validation and test splits.")
    except BaseException:
        for handle in handles.values():
            handle.close()
        index.close()
        shutil.rmtree(output)
        raise
    for handle in handles.values():
        handle.close()
    index.close()
    tokenizer.save_pretrained(output / "tokenizer")
    shutil.copyfile(ROOT / "TOKENIZER-LICENSE", output / "TOKENIZER-LICENSE")
    write_json(output / "corpus-manifest.json", {
        "schema_version": 1,
        "tokenizer": lock["tokenizer"],
        "architecture_lock_sha256": sha256_file(ROOT / "architecture.lock.json"),
        "source_files": {str(path.resolve()): sha256_file(path) for path in args.data},
        "record_counts": counts,
        "token_counts": token_counts,
        "token_files": {split: sha256_file(output / f"{split}.tokens") for split in SPLITS},
        "token_dtype": "uint32-little-endian",
        "unreviewed_records": unreviewed,
        "synthetic_records": synthetic,
        "leakage_checks": "Exact normalized duplicates, source groups, duplicate answers and sampled 3-word shingles; semantic leakage still requires human review.",
        "objective": "All-token causal language modeling over documents and complete formatted conversations, from scratch.",
    })
    print(f"Prepared private tokens in {output}. No neural weights were loaded or created.")


if __name__ == "__main__":
    main()
