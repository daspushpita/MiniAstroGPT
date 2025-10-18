import requests, time, json, xml.etree.ElementTree as ET

# ---------------------------
# PARAMETERS
# ---------------------------
# Known astro-ph categories (explicit OR instead of wildcard)
ASTRO_CATS = ["astro-ph", "astro-ph.HE", "astro-ph.CO", "astro-ph.EP", "astro-ph.GA",
              "astro-ph.IM", "astro-ph.SR"]
CAT_QUERY = "(" + " OR ".join(f"cat:{c}" for c in ASTRO_CATS) + ")"


date_from = "202501010000"  # 1 Jan 2025 00:00 UTC
date_to   = "202510172359"  # 17 Oct 2025 23:59 UTC
max_results = 200  # per request
total_expected = 16995       # for info
outfile = "astro_abstracts_2025.jsonl"

# ---------------------------
# MAIN LOOP
# ---------------------------
base_url = "http://export.arxiv.org/api/query"
start = 0
count = 0
with open(outfile, "w", encoding="utf-8") as fout:
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
        response = requests.get(base_url, params=params)
        print("URL sent:", response.url)
        # Parse the XML response
        root = ET.fromstring(response.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entries = root.findall("atom:entry", ns)
        print("entries on this page:", len(entries))

        if not entries:
            print("No more entries.")
            break        
        for e in entries:
            title = e.find("atom:title", ns).text.strip().replace("\n", " ")
            summary = e.find("atom:summary", ns).text.strip().replace("\n", " ")
            paper_id = e.find("atom:id", ns).text.strip()
            published = e.find("atom:published", ns).text.strip()
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
