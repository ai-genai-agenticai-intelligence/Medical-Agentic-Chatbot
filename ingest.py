import argparse
import hashlib
from pathlib import Path
import re
import time
import pymupdf as fitz
from app.knowledge import ensure_index, embed


def chunks(text, size=1100, overlap=180):
    text = ' '.join(text.split())
    step = size - overlap
    return [text[i:i+size] for i in range(0, len(text), step) if len(text[i:i+size]) > 80]


def embed_with_retry(texts, max_retries=8):
    for attempt in range(max_retries):
        try:
            return embed(texts)
        except Exception as e:
            err_msg = str(e)
            if "RESOURCE_EXHAUSTED" in err_msg or "429" in err_msg or "quota" in err_msg.lower():
                delay = 25 * (attempt + 1)
                match = re.search(r"retryDelay':\s*'(\d+)s'", err_msg)
                if match:
                    delay = int(match.group(1)) + 3
                print(f"[Rate limit hit] Pausing for {delay}s before retry (attempt {attempt+1}/{max_retries})...", flush=True)
                time.sleep(delay)
            else:
                raise
    raise RuntimeError("Embedding failed after maximum retries due to persistent rate limiting.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf', default='data/Medical_book.pdf')
    parser.add_argument('--start-page', type=int, default=1, help='Page number to start indexing from (1-indexed)')
    parser.add_argument('--batch-size', type=int, default=30, help='Number of chunks per embedding batch')
    parser.add_argument('--delay', type=float, default=2.0, help='Seconds to wait between batches to avoid rate limits')
    args = parser.parse_args()
    pdf = Path(args.pdf)
    if pdf.is_dir():
        pdfs = list(pdf.glob('*.pdf'))
        if not pdfs:
            raise SystemExit(f'No PDF found in directory: {pdf}')
        pdf = pdfs[0]
    elif not pdf.exists():
        raise SystemExit(f'PDF not found: {pdf}')

    index = ensure_index()
    with fitz.open(pdf) as doc:
        batch = []
        total_indexed = 0
        for page_number, page in enumerate(doc, start=1):
            if page_number < args.start_page:
                continue

            for part, content in enumerate(chunks(page.get_text()), start=1):
                key = hashlib.sha256(f'{pdf.name}:{page_number}:{part}:{content}'.encode()).hexdigest()
                batch.append((key, content, {'page': page_number, 'text': content, 'file': pdf.name}))
                if len(batch) >= args.batch_size:
                    vectors = embed_with_retry([b[1] for b in batch])
                    index.upsert(vectors=[{'id': b[0], 'values': v, 'metadata': b[2]} for b, v in zip(batch, vectors)])
                    total_indexed += len(batch)
                    batch = []
                    time.sleep(args.delay)

            if page_number % 25 == 0:
                print(f'Processed pages: {page_number}/{len(doc)} | Total chunks indexed: {total_indexed}', flush=True)

        if batch:
            vectors = embed_with_retry([b[1] for b in batch])
            index.upsert(vectors=[{'id': b[0], 'values': v, 'metadata': b[2]} for b, v in zip(batch, vectors)])
            total_indexed += len(batch)

    print(f'Ingestion complete. Total chunks indexed: {total_indexed}')


if __name__ == '__main__':
    main()