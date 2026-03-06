"""Fetch astrophysics abstracts from arXiv."""
import requests, time, json, xml.etree.ElementTree as ET
import glob, re
from pathlib import Path


class ArxivDownloader:
    
    def __init__(
        self,
        *,
        date_from: str,
        date_to: str,
        outfile: Path,
        max_results: int = 500,
        categories: list[str] | None = None,
        sleep_seconds: int = 10):
        
        
        self.date_from = date_from
        self.date_to = date_to
        self.outfile = Path(outfile)
        self.max_results = max_results
        self.sleep_seconds = sleep_seconds
        # ---------------------------
        # PARAMETERS
        # ---------------------------
        # Known astro-ph categories (explicit OR instead of wildcard)
        self.ASTRO_CATS = ["astro-ph", "astro-ph.HE", "astro-ph.CO", "astro-ph.EP", "astro-ph.GA",
            "astro-ph.IM", "astro-ph.SR"]
        self.CAT_QUERY = "(" + " OR ".join(f"cat:{c}" for c in self.ASTRO_CATS) + ")"


    def get_text(self, element, default=""):
        """Helper function to get text from the XML element.
        """
        if element is not None and element.text is not None:
            return element.text.strip().replace("\n", " ")
        return default

    # ---------------------------
    # MAIN LOOP
    # ---------------------------

    def download(self, *args, **kwargs):

        base_url = "https://export.arxiv.org/api/query"
        HEADERS = {
            "User-Agent": "AstroGPT-arxiv-scraper/0.1 (pushpitads1996@gmail.com)"
        }

        if self.outfile.exists():
            print(f"Output file {self.outfile} already exists. Appending...")
            existing_count = sum(1 for _ in self.outfile.open("r", encoding="utf-8"))
            start = existing_count
            count = existing_count
            mode = "a"
            print(f"Resuming from record {start}...")
        else:
            self.outfile.parent.mkdir(parents=True, exist_ok=True)
            start = 0
            count = 0
            mode = "w"
            print(f"Creating new output file {self.outfile}...")

        total_results = None

        with self.outfile.open(mode, encoding="utf-8") as fout:
            while True:
                print(f"Fetching results {start} to {start + self.max_results}...")
                params = {
                    "search_query": f"{self.CAT_QUERY} AND submittedDate:[{self.date_from} TO {self.date_to}]",
                    "start": start,
                    "max_results": self.max_results,
                    "sortBy": "submittedDate",
                    "sortOrder": "ascending"
                }
                # Send the request to arXiv’s server.
                # This line actually "asks" arXiv for the data.
                print(f"Fetching {start}-{start+self.max_results}...")
                
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
                    title = self.get_text(e.find("atom:title", ns))
                    summary = self.get_text(e.find("atom:summary", ns))
                    paper_id = self.get_text(e.find("atom:id", ns))
                    published = self.get_text(e.find("atom:published", ns))
                    cats = [c.attrib.get("term", "") for c in e.findall("atom:category", ns)]
                    
                    record = {"id": paper_id,
                                "title": title,
                                "abstract": summary,
                                "published": published,
                                "categories": cats}
                    
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1
                if len(entries) < self.max_results:
                    break
                start += self.max_results
                time.sleep(self.sleep_seconds)  # be polite to arXiv's servers
                
        print(f"Downloaded {count} abstracts to {self.outfile}.")
            
            
class Clean_Jsonl_Files:
    
    """Generating cleaned jsonl files
    """
    def __init__(self, INPUT_PATTERN="../raw/*.jsonl", 
                 MERGED_PATH=Path("../processed/all_raw.jsonl"), 
                 CLEANED_PATH=Path("../processed/all_clean.jsonl")):
        # Accept either str or Path for patterns/paths.
        self.INPUT_PATTERN = str(INPUT_PATTERN)
        self.MERGED_PATH = Path(MERGED_PATH)
        self.CLEANED_PATH = Path(CLEANED_PATH)
        self.MERGED_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.CLEANED_PATH.parent.mkdir(parents=True, exist_ok=True)
        
    def merge_inputs(self):
        with self.MERGED_PATH.open("w", encoding="utf-8") as fout:
            for path in sorted(glob.glob(self.INPUT_PATTERN)):
                with open(path, "r", encoding="utf-8") as fin:
                    for line in fin:
                        line = line.strip()
                        if line:
                            fout.write(line + "\n")

        print(f"Concatenated into {self.MERGED_PATH}")
        return fout

    def clean_text(self, t: str) -> str:
        if not t:
            return ""
        # remove inline LaTeX math like $...$
        t = re.sub(r"\$.*?\$", " ", t)
        # remove simple LaTeX commands like \alpha, \textbf{...}, \cite{...}
        t = re.sub(r"\\[a-zA-Z]+(\[[^\]]*\])?(\{[^}]*\})?", " ", t)
        # collapse whitespace
        t = re.sub(r"\s+", " ", t).strip()
        return t

    def clean_merged_file(self):
        with self.MERGED_PATH.open("r", encoding="utf-8") as fin, \
             self.CLEANED_PATH.open("w", encoding="utf-8") as fout:

            for line in fin:
                obj = json.loads(line)
                if "title" in obj:
                    obj["title_clean"] = self.clean_text(obj["title"])
                if "abstract" in obj:
                    obj["abstract_clean"] = self.clean_text(obj["abstract"])
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")