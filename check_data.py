import os

with open("22_datasets.qmd", "r", encoding="utf-8") as f:
    source = f.read()

data_files = [f for f in os.listdir("data") if os.path.isfile(os.path.join("data", f))]
data_files = [x for x in data_files if x != ".DS_Store"]

present = [f for f in data_files if f in source]
missing = [f for f in data_files if f not in source]

print("Number of matching files : {0:d}".format(len(present)))
print("Number of missing files : {0:d}".format(len(missing)))

if missing:
    print("\nMissing files remaining...")
    for f in sorted(missing):
        print(f)
else:
    print("\nAll files are good!")