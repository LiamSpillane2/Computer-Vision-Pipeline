import os

imgs = []
lbls = []
for dir, _, files in os.walk(r"..\data\weld_data"):
    if len(files) > 0:
        for file in files:
            if file.lower().endswith(".jpg"):
                imgs.append(file[:-4])
            elif file.lower().endswith(".txt"):
                lbls.append(file[:-4])

diff = set(imgs) ^ set(lbls)

with open("../data/missing_weld_labels.txt", "w") as f:
    for file in diff:
        f.write(f"{file}.jpg\n")
