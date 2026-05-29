import gzip
import json
from pathlib import Path
from multiprocessing import Pool, cpu_count


def read_file_docs(gz_file):
    docs = []
    with gzip.open(gz_file, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                json.loads(line) 
                docs.append(line.strip())
            except json.JSONDecodeError:
                continue
    return docs


def main():
    base_dir = Path("INPUR DIR")
    output_dir = Path("OUTPUT DIR")
    output_dir.mkdir(exist_ok=True)

    batch_size = 30000 # MAX_LINE
    batch = []
    file_count = 0
    total_docs = 0
    print(base_dir)
    json_files = list(base_dir.glob("CC-MAIN-*/segments/*/wet_compressed/*.json.gz")) # For Common Crawl dataset
    print(f"Total {len(json_files)} json files.")

    num_workers = min(cpu_count(), 8)

    with Pool(num_workers) as pool:
        for docs in pool.imap_unordered(read_file_docs, json_files, chunksize=100):
            for doc in docs:
                batch.append(doc)
                total_docs += 1

                if len(batch) == batch_size:
                    out_file = output_dir / f"batch_{file_count:07d}.jsonl"
                    with open(out_file, "w", encoding="utf-8") as out_f:
                        out_f.write("\n".join(batch) + "\n")
                    print(f"Saved {len(batch)} docs to {out_file}")
                    batch.clear()
                    file_count += 1

    if batch:
        out_file = output_dir / f"batch_{file_count:07d}.jsonl"
        with open(out_file, "w", encoding="utf-8") as out_f:
            out_f.write("\n".join(batch) + "\n")
        print(f"Saved remaining {len(batch)} docs to {out_file}")
        file_count += 1

    print("# Total Docs:", total_docs)
    print("# Total batch files :", file_count)


if __name__ == "__main__":
    main()
