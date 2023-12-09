import os
import tqdm.autonotebook as tqdm
import xml.etree.ElementTree as ET

step = 250
filename = "../datasets/stackoverflow/Comments.xml"
comments = []


context = ET.iterparse(filename, events=("start", "end"))
context = iter(context)
event, root = next(context)

progress = tqdm.tqdm(total=90000000, desc="Comments", position=0, leave=True)

i = 0
for event, elem in context:
    if event == "end" and elem.tag == "row":
        i += 1
        if i % step == 0:
            comments.append(elem.get("Text"))
            progress.update(step)

        elem.clear()

del context

print(f"Number of comments: {len(comments):,}")

save_dir = "../datasets/stackoverflow/other/comments"
os.makedirs(save_dir, exist_ok=True)

for i, comment in enumerate(tqdm.tqdm(comments)):
    with open(os.path.join(save_dir, f"{i}.txt"), "w") as f:
        f.write(comment)

print("Done!")
