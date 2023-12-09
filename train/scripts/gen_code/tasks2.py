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


def extract_code(text):
    code_blocks = re.findall(r"```.*?```", text, re.DOTALL)
    if not code_blocks:
        return None
    all_code_blocks = []
    for i, code_block in enumerate(code_blocks):
        lines = code_block.splitlines()
        lines[0] = lines[0].replace("```", "")
        if len(lines[0]) < 15:
            lines = lines[1:]
        if not lines:
            continue
        lines[-1] = lines[-1].replace("```", "")
        all_code_blocks.append("\n".join(lines))

    code = "\n".join(all_code_blocks)
    code = code.replace("Output:", "").replace("\n\n", "\n")
    return code


def ask_llama(prompt, context=None):
    url = "http://localhost:11433/api/generate"

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
    lang, alang, hint = languages[lang_name]

    constructs = utils.lang_constructs.lang_keywords[lang_name]
    all_kws = constructs
    all_kws = [v for v in list(set(all_kws)) if v.isalpha()]

    if lang_name in utils.lang_constructs.lang_libs:
        libs = utils.lang_constructs.lang_libs[lang_name]
        all_kws += [al for _, al, _ in libs]
        for _, _, kw in libs:
            all_kws += kw
        all_kws = [v for v in list(set(all_kws)) if v.isalpha()]

    all_requests = []
    for i in range(limit):
        keywords = ", ".join([f'"{s}"' for s in random.sample(all_kws, 2)])

        prompt = (
            base_prompt.replace("<lang>", lang)
            .replace("<keywords>", keywords)
            .replace("<alang>", alang)
            .replace("<hint>", hint),
        )
        request = prompt
        all_requests.append(request)

    if save_root is not None:
        save_dir = os.path.join(save_root, lang_name)
        os.makedirs(save_dir, exist_ok=True)

    code_blocks = []

    for i, prompt in enumerate(
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

        code = extract_code(response)
        if code is None:
            continue
        code_blocks.append(code)

        if save_root is not None:
            with open(os.path.join(request_save_dir, "0.txt"), "w") as f:
                f.write(code)

    return code_blocks


base_prompt = """
Write me an example of working with <lang> language. Use some of the following keywords: <keywords>.
No needs to make it correct. Just make it look like a real code.
Write just a code without explanation and make it very natural.
WRAP ALL COMMANDS WITH TRIPLE BACKTICKS LIKE THESE:``` ```.
USE SPECIFIC FOR <lang> CONSTRUCTS. <hint>
<hint>
DON'T USE COMMENTS IN THE CODE. DON'T COMMENT THE CODE. ANSWER WITH ONE CODE BLOCK.
"""

languages = {
    # "TYPESCRIPT": [
    #     "TypeScript",
    #     "MAKE IT CLEAR THAT IT IS NOT A JavaScript.",
    #     "USE TYPESCRIPT SPECIFIC DEFENITIONS (:type), CLASSES AND ENUMS.",
    # ],
    # "JAVASCRIPT": [
    #     "JavaScript",
    #     "MAKE IT CLEAR THAT IT IS NOT A TypeScript.",
    #     "USE JAVASCRIPT SPECIFIC VARIABLES DEFENITION AND FUNCTIONS.",
    # ],
    # "C": ["C", "MAKE IT CLEAR THAT IT IS NOT A C++", ""],
    # "CPLUSPLUS": ["C++", "MAKE IT CLEAR THAT IT IS NOT A C lanuage.", ""],
    # "SHELL": ["Bash (Unix Shell)", "WRITE ONLY ONE LINE COMMAND.", ""],
    # "PYTHON": ["python", "WRITE THE ONLY ONE LINE OF CODE!", ""],
    "SQL": ["SQL", "", ""],
}

count_offsets = {
    # "TYPESCRIPT": [3000, 5000],
    "JAVASCRIPT": [5000, 5000],
    "SHELL": [4000, 5000],
    "PYTHON": [3000, 4000],
    "SQL": [5000, 0],
}

if __name__ == "__main__":
    all_langs = list(languages.keys())
    for lang in all_langs:
        limit, offset = count_offsets[lang]

        res = get_code(
            lang, limit=limit, offset=offset, save_root="../../datasets/llama/tasks2/"
        )
