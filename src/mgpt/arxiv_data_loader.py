from data_modules import ArxivDownloader, split_data_class, jason_to_txt

downloader = ArxivDownloader()
count, path = downloader.download_abstracts()
print("Returned:", count, "File:", path)

path = "/Users/pushpita/Documents/ML Projects/Building_LLM_from_scratch/MiniAstroLM/data/astro_abstracts_2025.jsonl"
split_data = split_data_class(path, split_ratio=0.1, seed=42).split_data()
print("Data split into:", split_data)

convert_data_train = jason_to_txt("train_data.jsonl", "train_data.txt").convert()
convert_data_val = jason_to_txt("val_data.jsonl", "val_data.txt").convert()