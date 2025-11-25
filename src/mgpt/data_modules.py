import requests, time, json, xml.etree.ElementTree as ET
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split

class ArxivDownloader:
    
    def __init__(self):
        # ---------------------------
        # PARAMETERS
        # ---------------------------
        # Known astro-ph categories (explicit OR instead of wildcard)
        self.ASTRO_CATS = ["astro-ph", "astro-ph.HE", "astro-ph.CO", "astro-ph.EP", "astro-ph.GA",
                    "astro-ph.IM", "astro-ph.SR"]
        self.CAT_QUERY = "(" + " OR ".join(f"cat:{c}" for c in self.ASTRO_CATS) + ")"


        self.date_from = "201501010000"  # 1 Jan 2015 00:00 UTC
        self.date_to   = "202510172359"  # 17 Oct 2025 23:59 UTC
        self.max_results = 200  # per request
        self.total_expected = 20000       # for info
        self.outfile = "astro_abstracts_last10_years.jsonl"

        # ---------------------------
        # MAIN LOOP
        # ---------------------------
        self.base_url = "http://export.arxiv.org/api/query"
        self.start = 0
        self.count = 0
        
    
    def download_abstracts(self):
        
        start_dt = pd.to_datetime(self.date_from, format="%Y%m%d%H%M")
        end_dt   = pd.to_datetime(self.date_to, format="%Y%m%d%H%M")
        months = pd.date_range(start_dt, end_dt, freq="MS").to_pydatetime().tolist() + [end_dt.to_pydatetime()]
        
        with open(self.outfile, "w", encoding="utf-8") as fout:
            for i in range(len(months) - 1):   
                d1 = months[i].strftime("%Y%m%d%H%M")
                d2 = months[i + 1].strftime("%Y%m%d%H%M")
                print(f"\n {d1} to {d2}")
                self.start = 0  
                while True:
                    print(f"Fetching results {self.start} to {self.start + self.max_results}...")
                    params = {
                        "search_query": f"{self.CAT_QUERY} AND submittedDate:[{d1} TO {d2}]",
                        "start": self.start,
                        "max_results": self.max_results,
                        "sortBy": "submittedDate",
                        "sortOrder": "ascending"
                    }
                    # Send the request to arXiv’s server.
                    # This line actually "asks" arXiv for the data.
                    print(f"Fetching {self.start}-{self.start+self.max_results}...")
                    response = requests.get(self.base_url, params=params)
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
                        self.count += 1
                    if len(entries) < self.max_results:
                        break
                    self.start += self.max_results
                    time.sleep(3)  # be polite to arXiv's servers
        print(f"Downloaded {self.count} abstracts to {self.outfile}.")
        return self.count, self.outfile
            
            
class split_data_class:
    
    def __init__(self, input_jason, split_ratio, seed=42):
        self.input_jason = input_jason
        self.split_ratio = split_ratio
        self.seed = seed
        
    def split_data(self):
        """Splits the data into train and validation sets

        Args:
            input_jason (_type_): input jason file path
            split_ratio (_type_): ratio for train and validation split
            seed (int, optional): _description_. Defaults to 42.
        """
        df = pd.read_json(self.input_jason, lines=True, orient="records")
        train_df, val_df = train_test_split(df, test_size=self.split_ratio, random_state=self.seed)

        train_df.to_json("train_data.jsonl", lines=True, orient="records", force_ascii=False)
        val_df.to_json("val_data.jsonl", lines=True, orient="records", force_ascii=False)
        print(f"Train: {len(train_df)}, Val: {len(val_df)} saved to JSONL files")
        return train_df, val_df
    
class jason_to_txt:
    
    def __init__(self, jason_file, output_txt_file):
        self.jason_file = jason_file
        self.output_txt_file = output_txt_file
        
    def convert(self):
        df = pd.read_json(self.jason_file, lines=True, orient="records")
        print(f"Loaded {len(df)} entries from {self.jason_file}")
        
        with open(self.output_txt_file, "w", encoding="utf-8") as fout:
            for abstract in df["abstract"]:
                abstract = abstract.strip().replace("\n", " ")
                fout.write(abstract + "\n\n<eos>\n\n")  # separate samples
        print(f"Saved abstracts to {self.output_txt_file}")
        return self.output_txt_file
