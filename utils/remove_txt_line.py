import os

for dir, _, files in os.walk(r"..\data\license_plate_detection"):
    if len(files) > 0:
        for file in files:
            if file.lower().endswith(".txt"):
                try:
                    *p, = open(rf"{dir}\{file}")
                    if len(p) > 0 and p[0].startswith("class"):
                        del p[0]
                    with open(rf"{dir}\{file}", "w") as f:
                        for line in p:
                            f.write(line)
                except Exception as e:
                    print(file)