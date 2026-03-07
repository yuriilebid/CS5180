#-------------------------------------------------------------
# AUTHOR: Yurii Lebid
# FILENAME: SPIMP_index.py
# SPECIFICATION: Build SPIMI-based inverted index pipeline
# FOR: CS 5180- Assignment #2
# TIME SPENT: 5 days in total (~2 hours of coding)
#-----------------------------------------------------------*/

import pandas as pd
import heapq
from sklearn.feature_extraction.text import CountVectorizer

INPUT_PATH = "corpus.tsv"
BLOCK_SIZE = 100
NUM_BLOCKS = 10

READ_BUFFER_LINES_PER_FILE = 100
WRITE_BUFFER_LINES = 500


def build_blocks():

    chunk_iter = pd.read_csv(
        INPUT_PATH,
        sep="\t",
        names=["doc_id", "text"],
        chunksize=BLOCK_SIZE,
        encoding="utf-8"
    )

    block_id = 1

    for chunk in chunk_iter:

        vectorizer = CountVectorizer(stop_words='english')
        X = vectorizer.fit_transform(chunk["text"])

        terms = vectorizer.get_feature_names_out()
        index = {}

        for term_idx, term in enumerate(terms):
            doc_indices = X[:, term_idx].nonzero()[0]
            postings = sorted(
                int(chunk.iloc[i]["doc_id"].replace("D", ""))
                for i in doc_indices
            )
            if postings:
                index[term] = postings

        filename = f"block_{block_id}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            for term in sorted(index.keys()):
                postings_str = ",".join(map(str, index[term]))
                f.write(f"{term}:{postings_str}\n")

        block_id += 1


def merge_blocks():

    files = [open(f"block_{i}.txt", encoding="utf-8") for i in range(1, NUM_BLOCKS+1)]

    buffers = []
    pointers = []
    heap = []

    for i, f in enumerate(files):
        lines = []
        for _ in range(READ_BUFFER_LINES_PER_FILE):
            line = f.readline()
            if not line:
                break
            lines.append(line.strip())
        buffers.append(lines)
        pointers.append(0)

        if lines:
            term = lines[0].split(":")[0]
            heapq.heappush(heap, (term, i))

    write_buffer = []

    with open("final_index.txt", "w", encoding="utf-8") as out:

        while heap:

            current_term, file_idx = heapq.heappop(heap)
            merged_postings = set()

            involved_files = [file_idx]

            while heap and heap[0][0] == current_term:
                _, idx = heapq.heappop(heap)
                involved_files.append(idx)

            for idx in involved_files:
                line = buffers[idx][pointers[idx]]
                postings = list(map(int, line.split(":")[1].split(",")))
                merged_postings.update(postings)

                pointers[idx] += 1

                if pointers[idx] >= len(buffers[idx]):
                    buffers[idx] = []
                    pointers[idx] = 0
                    for _ in range(READ_BUFFER_LINES_PER_FILE):
                        new_line = files[idx].readline()
                        if not new_line:
                            break
                        buffers[idx].append(new_line.strip())

                if buffers[idx] and pointers[idx] < len(buffers[idx]):
                    next_term = buffers[idx][pointers[idx]].split(":")[0]
                    heapq.heappush(heap, (next_term, idx))

            merged_list = sorted(merged_postings)
            line = f"{current_term}:{','.join(map(str, merged_list))}\n"
            write_buffer.append(line)

            if len(write_buffer) >= WRITE_BUFFER_LINES:
                out.writelines(write_buffer)
                write_buffer = []

        if write_buffer:
            out.writelines(write_buffer)

    for f in files:
        f.close()


if __name__ == "__main__":
    build_blocks()
    merge_blocks()