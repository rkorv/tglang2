# Out of the scope:
# TGLANG_LANGUAGE_FUNC
# TGLANG_LANGUAGE_TL

import requests
import json
import random
import tqdm
import re
import os

import sys

sys.path.append(".")
import utils.lang_constructs


def extract_code(text, start_symb):
    code_blocks = re.findall(r"```.*?```", text, re.DOTALL)
    if not code_blocks:
        return None
    all_code_blocks = []
    for i, code_block in enumerate(code_blocks):
        if i == 0 and start_symb not in code_block:
            break
        lines = code_block.splitlines()
        lines[0] = lines[0].replace("```", "")
        lines[-1] = lines[-1].replace("```", "")
        all_code_blocks.append("\n".join(lines))

    code = "\n".join(all_code_blocks)
    if start_symb not in code:
        return None
    code = code.replace("Output:", "").replace("\n\n", "\n")

    if start_symb == ">>>" and "$" in code:
        return None

    return code


def ask_llama(prompt, context=None):
    url = "http://localhost:11434/api/generate"

    req = {"model": "llama2", "stream": False, "prompt": prompt}
    if context is not None:
        req["context"] = context

    while True:
        try:
            response = requests.request(
                "POST", url, data=json.dumps(req), timeout=60
            ).json()
            break
        except:
            continue

    return response["response"], response["context"]


def get_code(lang_name, limit=1000, offset=0, save_root=None):
    lang, start_symbs = languages[lang_name]

    constructs = utils.lang_constructs.lang_keywords[lang_name]
    all_kws = constructs
    all_kws = [v for v in list(set(all_kws)) if v.isalpha()]

    if lang_name == "SQL":
        all_kws += ["+", "-", "*", "/", "%"]
    else:
        libs = utils.lang_constructs.lang_libs[lang_name]
        all_kws += [al for _, al, _ in libs]
        all_kws = [v for v in list(set(all_kws)) if v.isalpha()]

    all_requests = []
    for i in range(limit):
        num_commands = random.randint(2, 4)
        keywords = ", ".join([f'"{s}"' for s in random.sample(all_kws, 3)])
        start_symb = random.choice(start_symbs)

        prompt = (
            base_prompt.replace("<lang>", lang)
            .replace("<keywords>", keywords)
            .replace("<num_commands>", str(num_commands))
            .replace("<start_symb>", start_symb),
        )
        request = (prompt, start_symb)
        all_requests.append(request)

    if save_root is not None:
        save_dir = os.path.join(save_root, lang_name)
        os.makedirs(save_dir, exist_ok=True)

    code_blocks = []

    for i, (prompt, start_symb) in enumerate(
        tqdm.tqdm(all_requests, desc="Processing language '%s'" % lang_name)
    ):
        real_i = i + offset
        if save_root is not None:
            request_save_dir = os.path.join(save_dir, str(real_i))
            if os.path.exists(request_save_dir) and len(
                os.listdir(request_save_dir)
            ) == len(request):
                continue

            os.makedirs(request_save_dir, exist_ok=True)

        response, _ = ask_llama(prompt[0])

        code = extract_code(response, start_symb)
        if code is None:
            continue
        code_blocks.append(code)

        if save_root is not None:
            with open(os.path.join(request_save_dir, "0.txt"), "w") as f:
                f.write(code)

    return code_blocks


base_prompt = """
Write me an example of working with <lang> commands in terminal (or in interpretator). Use some of the following keywords: <keywords>.
Write example of terminal output. Start your code with "<start_symb>". And continue with the output.
No needs to make it correct. Just make it look like a real code.
Wrap full text in a brackets. Example:```
$ <command_1>
<output_1>
$ <command_N>
<output_N>
```
Write not more than <num_commands> commands and outputs.
Write just a code without explanation and make it very natural.
WRAP ALL COMMANDS WITH TRIPLE BACKTICKS LIKE THESE:``` ```.
DON'T USE COMMENTS IN THE CODE. DON'T COMMENT THE CODE. ANSWER WITH ONE CODE BLOCK.
"""

languages = {
    # "SQL": ["SQL", ["sql>"]],
    # "SHELL": ["Bash (Unix shell)", ["$"]],
    "PYTHON": ["Python", [">>>"]],
}

if __name__ == "__main__":
    all_langs = list(languages.keys())
    for lang in all_langs:
        res = get_code(
            lang, limit=4000, offset=0, save_root="../../datasets/llama/shell/"
        )
