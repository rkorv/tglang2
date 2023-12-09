import os
import gc
import tqdm.autonotebook as tqdm
import xml.etree.ElementTree as ET
import html
import xml.etree.ElementTree as ET
import re

step = 100
total = 1000000

MIN_CODE_LEN = 50

filename = "../datasets/stackoverflow/Posts.xml"


def extract_code_and_other_from_body(body_text):
    body_text = html.unescape(body_text)
    pre_code_blocks = re.findall(r"<pre><code>(.*?)</code></pre>", body_text, re.DOTALL)
    body_without_pre_code = re.sub(
        r"<pre><code>.*?</code></pre>", "", body_text, flags=re.DOTALL
    )
    inline_code_blocks = re.findall(r"<code>([^<]+)</code>", body_without_pre_code)
    non_code_text = re.sub(r"<code>[^<]+</code>", "", body_without_pre_code)
    non_code_text = re.sub(r"<[^>]+>", "", non_code_text).strip()
    all_code_blocks = pre_code_blocks + inline_code_blocks
    return all_code_blocks, non_code_text


def extract_tags(tags_str):
    return set(tags_str[1:-1].split("><"))


progress = tqdm.tqdm(total=total, desc="Posts", position=0, leave=True)

# all_code = []
# all_tags = []
# all_other = []
i = 0

context = ET.iterparse(filename, events=("start", "end"))
context = iter(context)
event, root = next(context)

save_dir = "../datasets/stackoverflow/other/posts/0"
os.makedirs(save_dir, exist_ok=True)

for event, elem in context:
    if event == "end" and elem.tag == "row":
        i += 1

        if i % step == 0:
            body = elem.attrib.get("Body", "")
            tags_str = elem.attrib.get("Tags", "")
            tags = extract_tags(tags_str)
            code_blocks, other_text = extract_code_and_other_from_body(body)
            if other_text:
                with open(os.path.join(save_dir, f"{i}.txt"), "w") as f:
                    f.write(other_text)
                progress.update(1)

        elem.clear()

        if i >= total * step:
            break

print("Done!")
