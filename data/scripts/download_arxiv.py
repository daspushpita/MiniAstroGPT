import os
import requests, time, json, xml.etree.ElementTree as ET
import argparse
# ---------------------------
# PARAMETERS
# ---------------------------
# Known astro-ph categories (explicit OR instead of wildcard)
ASTRO_CATS = ["astro-ph", "astro-ph.HE", "astro-ph.CO", "astro-ph.EP", "astro-ph.GA",
              "astro-ph.IM", "astro-ph.SR"]
CAT_QUERY = "(" + " OR ".join(f"cat:{c}" for c in ASTRO_CATS) + ")"


# date_from = "202501010000"  # 1 Jan 2025 00:00 UTC
# date_to   = "202510172359"  # 17 Oct 2025 23:59 UTC
# max_results = 200  # per request
# total_expected = 16995       # for info
# outfile = "astro_abstracts_2025.jsonl"

def parse_args():
    parser = argparse.ArgumentParser(description="Download arXiv abstracts for astro-ph category.")
    parser.add_argument(
        "--date-from",
        default="202501010000",
        help="Start submittedDate in yyyymmddhhMM format (default: 202501010000).",
    )
    
    parser.add_argument(
        "--date-to",
        default="202510172359",
        help='End submittedDate in yyyymmddhhMM format (default: 202510172359).',
    )
    parser.add_argument(
        "--outfile",
        default="astro_abstracts_2025.jsonl",
        help="Output file name (default: astro_abstracts_2025.jsonl).",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=500,
        help="Number of results per request (default: 200, max allowed by arXiv is 2000).",
    )

    return parser.parse_args()
    
def get_text(element, default=""):
    """Helper function to get text from the XML element.
    """
    if element is not None and element.text is not None:
        return element.text.strip().replace("\n", " ")
    return default
# ---------------------------
# MAIN LOOP
# ---------------------------

def main(*args, **kwargs):
    args = parse_args()
    date_from = args.date_from
    date_to = args.date_to
    outfile = args.outfile
    max_results = args.max_results

    base_url = "https://export.arxiv.org/api/query"
    HEADERS = {
        "User-Agent": "AstroGPT-arxiv-scraper/0.1 (pushpitads1996@gmail.com)"
    }

    if os.path.exists(outfile):
        print(f"Output file {outfile} already exists. Writing to it...")
        existing_count = 0
        with open(outfile, "r", encoding="utf-8") as f:
            for _ in f:
                existing_count += 1
        start = existing_count
        count = existing_count
        mode = "a"
        print(f"Resuming record from {start}...")
    else:
        mode = "w"
        start = 0
        count = 0
        print(f"Creating new output file {outfile}...")

    total_results = None
    with open(outfile, mode, encoding="utf-8") as fout:
        while True:
            print(f"Fetching results {start} to {start + max_results}...")
            params = {
                "search_query": f"{CAT_QUERY} AND submittedDate:[{date_from} TO {date_to}]",
                "start": start,
                "max_results": max_results,
                "sortBy": "submittedDate",
                "sortOrder": "ascending"
            }
            # Send the request to arXiv’s server.
            # This line actually "asks" arXiv for the data.
            print(f"Fetching {start}-{start+max_results}...")
            
            try:
                response = requests.get(base_url, params=params, headers=HEADERS, timeout=60)
                response.raise_for_status()
            except requests.RequestException as e:
                print(f"Request failed at start={start}: {e}")
                break

            print("URL sent:", response.url)
            # Parse the XML response
            
            try:
                root = ET.fromstring(response.text)
            except ET.ParseError as e:
                print(f"XML parse error at start={start}: {e}")
                break 

            ns = {
                "atom": "http://www.w3.org/2005/Atom",
                "opensearch": "http://a9.com/-/spec/opensearch/1.1/"
            }
            if total_results is None and start == 0:
                tr = root.find("opensearch:totalResults", ns)
                if tr is not None and tr.text is not None:
                    total_results = int(tr.text.strip())
                    print(f"arXiv reports total_results = {total_results}")

            entries = root.findall("atom:entry", ns)
            print("entries on this page:", len(entries))

            if not entries:
                print("No more entries.")
                break        
            for e in entries:
                title = get_text(e.find("atom:title", ns))
                summary = get_text(e.find("atom:summary", ns))
                paper_id = get_text(e.find("atom:id", ns))
                published = get_text(e.find("atom:published", ns))
                cats = [c.attrib.get("term", "") for c in e.findall("atom:category", ns)]
                
                record = {"id": paper_id,
                            "title": title,
                            "abstract": summary,
                            "published": published,
                            "categories": cats}
                
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
            if len(entries) < max_results:
                break
            start += max_results
            time.sleep(3)  # be polite to arXiv's servers
            
    print(f"Downloaded {count} abstracts to {outfile}.")
    
if __name__ == "__main__":
    main()