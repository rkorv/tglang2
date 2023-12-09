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

base_prompt = """
The TL (Type Language) used in Telegram's MTProto is structured as follows:
Overview: TL programs are divided into two main sections separated by the ---functions--- keyword. The first section declares built-in and aggregate types (constructors), while the second section comprises declared functions or functional combinators. Combinator declarations end with a semicolon. Additional type declarations post-function declaration use the ---types--- keyword. Functional combinators in the type section have their result type starting with an exclamation point​​.
Namespaces: TL utilizes composite constructions for namespace and identifier declarations, where the identifier part after the period follows a rule: uppercase for type identifiers and lowercase for constructor identifiers. No special declaration is required for namespaces​​.
Comments: The comment syntax in TL is similar to that in C++​​.
Example: TL language includes built-in types like int, long, double, string, and null. It also allows for the creation of custom types and constructors, such as pair, triple, and user, each with their own properties and structures​​.
This structure allows for robust and flexible data type definition and manipulation within the Telegram platform.

################### EXAMPLES ###################
// Built-In Types Definition
int#a8509bda ? = Int;
long ? = Long;
double ? = Double;
string ? = String;
null = Null;

// Custom Composite Types
vector {t:Type} # [ t ] = Vector t;
pair x:Object y:Object = Pair;
triple x:Object y:Object z:Object = Triple;

// User and Group Types
user#d23c81a3 id:int first_name:string last_name:string = User;
no_user#c67599d1 id:int = User;
group id:int title:string last_name:string = Group;
no_group = Group;

// Type Variables and Hash Types
intHash {alpha:Type} vector<coupleInt<alpha>> = IntHash<alpha>;
strHash {alpha:Type} (vector (coupleStr alpha)) = StrHash alpha;

// built-in types
int#a8509bda ? = Int;
long ? = Long;
double ? = Double;
string ? = String;
null = Null;

vector {t:Type} # [ t ] = Vector t;
coupleInt {alpha:Type} int alpha = CoupleInt<alpha>;
coupleStr {gamma:Type} string gamma = CoupleStr gamma;
/* The name of the type variable is irrelevant: "gamma" could be replaced with "alpha";
   However, the combinator number will depend on the specific choice. */

intHash {alpha:Type} vector<coupleInt<alpha>> = IntHash<alpha>;
strHash {alpha:Type} (vector (coupleStr alpha)) = StrHash alpha;
intSortedHash {alpha:Type} intHash<alpha> = IntSortedHash<alpha>;
strSortedHash {alpha:Type} (strHash alpha) = StrSortedHash alpha;

// instantiating polymorphic types ("templates")
// outdated, ignored altogether
Vector int;
// with syntactic sugar as well:
Vector<string>;
Vector Object;
IntHash int;
IntHash string;
IntHash Object;
StrHash int;
StrHash string;
StrHash Object;

// custom types
pair x:Object y:Object = Pair;
triple x:Object y:Object z:Object = Triple;

user#d23c81a3 id:int first_name:string last_name:string = User;
no_user#c67599d1 id:int = User;
group id:int title:string last_name:string = Group;
no_group = Group;

---functions---

// Maybe some built-in arithmetic functions; inverse quotes make "identifiers" out of arbitrary non-alphanumeric strings
`+` Int Int = Int;
`-` Int Int = Int;
`+` Double Double = Double;
// ...

// API functions (aka RPC functions)
getUser#b0f732d5 int = User;
getUsers#2d84d5f5 (Vector int) = Vector User;

################### TASK ###################

Here are a documentation and some examples of working with TL language.
Write your own example based on this language structure. Make it differ from original text.
Use some of the following keywords: <keywords>.
No needs to make it correct. Just make it look like a real code.
Write just a code without explanation and make it very natural.
WRAP ALL COMMANDS WITH TRIPLE BACKTICKS LIKE THESE:``` ```.
DON'T USE COMMENTS IN THE CODE. DON'T COMMENT THE CODE. ANSWER WITH ONE CODE BLOCK.
"""


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
        lines[-1] = lines[-1].replace("```", "")
        all_code_blocks.append("\n".join(lines))

    code = "\n".join(all_code_blocks)
    if "int main" in code:
        return None

    code = code.replace("\n\n", "\n")
    return code


def ask_llama(prompt, context=None):
    url = "http://localhost:11434/api/generate"

    req = {"model": "llama2:13b", "stream": False, "prompt": prompt}
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


def get_code(limit=1000, offset=0, save_root=None):
    lang_name = "TL"
    constructs = utils.lang_constructs.lang_keywords[lang_name]
    all_kws = constructs

    all_requests = []
    for i in range(limit):
        keywords = ", ".join([f'"{s}"' for s in random.sample(all_kws, 2)])
        request = base_prompt.replace("<keywords>", keywords)
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
            if os.path.exists(request_save_dir):
                continue

            os.makedirs(request_save_dir, exist_ok=True)

        response, _ = ask_llama(prompt)

        code = extract_code(response)
        if code is None:
            continue
        code_blocks.append(code)

        if save_root is not None:
            with open(os.path.join(request_save_dir, "0.txt"), "w") as f:
                f.write(code)

    return code_blocks


if __name__ == "__main__":
    res = get_code(limit=4000, offset=0, save_root="../../datasets/llama/tasks2/")
