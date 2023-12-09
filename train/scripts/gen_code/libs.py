# Out of the scope:
# TGLANG_LANGUAGE_FUNC
# TGLANG_LANGUAGE_TL

import requests
import json
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

    code_blocks = code_blocks[0]
    code_blocks = code_blocks.splitlines()
    code_blocks = "\n".join(code_blocks[1:-1])
    return code_blocks


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


def get_code(
    lang,
    reset_context=4,
    save_root=None,
):
    lang_name, hint, module_type = languages[lang]

    all_requests = []
    for data in utils.lang_constructs.lang_libs[lang]:
        module, _, kws = data
        request = [
            base_prompt_task.replace("<module>", module)
            .replace("<lang>", lang_name)
            .replace("<hint>", hint)
            .replace("<module_type>", module_type)
        ]
        for more_prompt in more_prompts_task:
            request.append(
                more_prompt.replace("<module>", module)
                .replace("<lang>", lang_name)
                .replace("<hint>", hint)
                .replace("<module_type>", module_type)
            )

        for kw in kws:
            request.append(
                use_keywords_task.replace("<keyword>", kw)
                .replace("<lang>", lang_name)
                .replace("<hint>", hint)
                .replace("<module_type>", module_type)
            )

        all_requests.append(request)

    code_blocks = []

    if save_root is not None:
        save_dir = os.path.join(save_root, lang)
        os.makedirs(save_dir, exist_ok=True)

    for i, request in enumerate(
        tqdm.tqdm(all_requests, desc="Processing language '%s'" % lang_name)
    ):
        if save_root is not None:
            request_save_dir = os.path.join(save_dir, str(i))
            if os.path.exists(request_save_dir) and len(
                os.listdir(request_save_dir)
            ) == len(request):
                continue

            os.makedirs(request_save_dir, exist_ok=True)

        context = None
        for j, prompt in enumerate(request):
            if j % reset_context == 0:
                context = None
            response, curr_context = ask_llama(prompt, context)
            context = curr_context

            code = extract_code(response)
            if code is None:
                continue
            code_blocks.append(code)

            if save_root is not None:
                with open(os.path.join(request_save_dir, "%d.txt" % j), "w") as f:
                    f.write(code)

    return code_blocks


base_prompt_task = """
We are writing dataset for  programming language identification.
Write me an example of <lang> language and you must use "<module>" <module_type> in your code to show an example.
Not necessary to make it work, but make sure that it is <lang> language. Just write a code that looks like a real code <hint>.
Wrap the code in a ```code block```. Provide the only one code block in your answer.
Write just a code without explanation and make it very natural.
DON'T USE COMMENTS IN THE CODE. DON'T COMMENT THE CODE.
"""

more_prompts_task = [
    """
DON'T REPEAT EXAMPLES ABOVE!
give me another example which doesn't look like a previous example and shows how to work with "<module>" in <lang> language!
Write just a code without explanation and make it very natural.
DON'T USE COMMENTS IN THE CODE. DON'T COMMENT THE CODE.
""",
]

use_keywords_task = """
DON'T REPEAT EXAMPLES ABOVE!
give me another example which doesn't look like a previous example and shows how to work with "<module>" in <lang> language!
USE WORD "<keyword>" IN THE CODE!
Wrap the code in a ```code block```.
DON'T USE COMMENTS IN THE CODE. DON'T COMMENT THE CODE.
"""


languages = {
    # programs
    "C": ["C", "(and it is clear that this is not C++)", "library"],
    "CPLUSPLUS": ["C++", "(and it is clear that this is not C)", "library"],
    "JAVA": ["Java", "", "library"],
    "CSHARP": ["C#", "", "library"],
    "OBJECTIVE_C": ["Objective-C", "", "library"],
    "PYTHON": ["Python", "", "library"],
    "RUBY": ["Ruby", "", "library"],
    "PHP": ["PHP", "(and it is clear that this is not Hack language)", "library"],
    "JAVASCRIPT": [
        "JavaScript",
        "(and it is clear that this is not TypeScript)",
        "library",
    ],
    "TYPESCRIPT": [
        "TypeScript",
        "(and it is clear that this is not JavaScript)",
        "library",
    ],
    "GO": ["Go", "", "library"],
    "RUST": ["Rust", "", "library"],
    "LUA": ["Lua", "", "library"],
    "SWIFT": ["Swift", "", "library"],
    "KOTLIN": ["Kotlin", "", "library"],
    "DART": ["Dart", "", "library"],
    "SHELL": ["Bash", "", "program"],
    "POWERSHELL": ["PowerShell", "", "program"],
}

if __name__ == "__main__":
    all_langs = list(languages.keys())
    for lang in all_langs:
        res = get_code(lang, save_root="../datasets/llama/libs/")
