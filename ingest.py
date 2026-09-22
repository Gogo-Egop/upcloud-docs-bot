import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from apify_client import ApifyClient
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

UPCLOUD = [
    "https://upcloud.com/docs/products/managed-postgresql/",
    "https://upcloud.com/docs/products/networking/",
]
POSTGRES = [
    "https://www.postgresql.org/docs/18/libpq-connect.html",
    "https://www.postgresql.org/docs/18/libpq-ssl.html",
]

client = ApifyClient(os.environ["APIFY_TOKEN"])
print("Collecting documentation with Apify...")
run = client.actor("apify/website-content-crawler").call(
    run_input={
        "startUrls": [{"url": url} for url in UPCLOUD + POSTGRES],
        "includeUrlGlobs": [url + "**" for url in UPCLOUD],
        "crawlerType": "playwright:adaptive",
        "maxCrawlPages": 100,
        "maxCrawlDepth": 5,
        "maxConcurrency": 3,
        "saveMarkdown": True,
        "respectRobotsTxtFile": True,
        "proxyConfiguration": {"useApifyProxy": True},
    },
    timeout_secs=1800,
)
if not run or run["status"] != "SUCCEEDED":
    raise RuntimeError("Crawl did not finish. Check the run in Apify Console.")

pages = {}
processed = 0
retrieved_at = datetime.now(timezone.utc).isoformat()
for item in client.dataset(run["defaultDatasetId"]).iterate_items():
    processed += 1
    crawl = item.get("crawl") or {}
    url = crawl.get("loadedUrl") or item.get("url", "")
    text = item.get("markdown") or item.get("text") or ""
    in_scope = url in POSTGRES or any(url.startswith(p) for p in UPCLOUD)
    if not in_scope or len(text.strip()) < 100 or (crawl.get("httpStatusCode") or 200) >= 400:
        continue
    title = (item.get("metadata") or {}).get("title") or url
    pages[url] = Document(
        page_content=f"{title}\n\n{text}",
        metadata={"source": url, "title": title, "retrieved_at": retrieved_at},
    )

if not all(any(url.startswith(prefix) for url in pages) for prefix in UPCLOUD):
    raise RuntimeError("An UpCloud section returned no usable pages. Check the crawl.")
if not all(url in pages for url in POSTGRES):
    raise RuntimeError("A PostgreSQL reference page was missing. Check the crawl.")
if processed >= 100:
    print("Page limit reached: this index may cover only part of the selected docs.")

chunks = RecursiveCharacterTextSplitter(
    chunk_size=3000, chunk_overlap=300
).split_documents(list(pages.values()))

namespace = "docs-" + uuid4().hex
store = PineconeVectorStore(
    index_name=os.environ["PINECONE_INDEX_NAME"],
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
    namespace=namespace,
)
print(f"Indexing {len(pages)} pages as {len(chunks)} passages...")
for start in range(0, len(chunks), 64):
    store.add_documents(chunks[start:start + 64])

temporary = ROOT / "namespace.tmp"
temporary.write_text(namespace, encoding="utf-8")
temporary.replace(ROOT / "namespace.txt")
print("Ready. Run: python -m streamlit run app.py")