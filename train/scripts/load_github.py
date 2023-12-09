import sys

sys.path.append(".")

import os
import subprocess
import requests
import uuid

import tqdm.autonotebook as tqdm
from utils.lang_enum import TGLANG_LANGUAGE_EXTENSIONS

GITHUB_API_URL = "https://api.github.com"
SAVE_ROOT = "/mnt/HDD8TB/workspace/comp/tg/datasets"
SAVE_DIR = os.path.join(SAVE_ROOT, "github")
SAVE_REPO_DIR = os.path.join(SAVE_ROOT, "github_repos")

PER_PAGE = 100
MAX_FILES_FROM_ONE_REPO = 100000  # unlimited
MAX_REPO_SIZE_KB = 100000  # 100 MB
MAX_FILE_SIZE_KB = 1000

TOKEN = "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"

HEADERS = {
    "Authorization": f"token {TOKEN}",
    "Accept": "application/vnd.github.v3+json",
}


def get_repositories_for_language(language, limit=10):
    repos = []
    page = 1
    while len(repos) < limit:
        l = f'"{language}"'
        response = requests.get(
            f"{GITHUB_API_URL}/search/repositories?q=language:{l}&per_page={PER_PAGE}&page={page}",
            headers=HEADERS,
        )
        items = response.json().get("items", [])
        if not items:
            break
        repos.extend(items[: limit - len(repos)])
        page += 1
    return repos


def get_repositories_for_query(language, limit=10):
    repos = []
    page = 1
    while len(repos) < limit:
        l = f"{language}"
        response = requests.get(
            f"{GITHUB_API_URL}/search/repositories?q={l}&per_page={PER_PAGE}&page={page}",
            headers=HEADERS,
        )
        items = response.json().get("items", [])
        if not items:
            break
        repos.extend(items[: limit - len(repos)])
        page += 1
    return repos


def count_files_in_directory(directory):
    """Recursively count files in a directory."""
    return sum([len(files) for _, _, files in os.walk(directory)])


def clone_and_process_repo(repo, language, extensions):
    author, repo_name = repo["full_name"].split("/")
    clone_dest_folder = os.path.join(SAVE_REPO_DIR, language, author, repo_name)
    files_dest_folder = os.path.join(SAVE_DIR, language, author, repo_name)

    if repo["size"] > MAX_REPO_SIZE_KB:
        return

    if not os.path.exists(clone_dest_folder):
        subprocess.run(
            ["git", "clone", repo["clone_url"], clone_dest_folder],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    for root, dirs, files in os.walk(clone_dest_folder):
        for filename in files:
            try:
                if any(
                    filename.lower().endswith(f"{ext.lower()}") for ext in extensions
                ):
                    source_filepath = os.path.join(root, filename)
                    if os.path.getsize(source_filepath) > (MAX_FILE_SIZE_KB * 1024):
                        continue
                    if language == "docker compose":
                        if "docker" not in os.path.basename(source_filepath).lower():
                            continue

                    dest_filepath = os.path.join(files_dest_folder, filename)
                    os.makedirs(os.path.dirname(dest_filepath), exist_ok=True)
                    if os.path.exists(dest_filepath):
                        basename = os.path.basename(dest_filepath)
                        basename, ext = os.path.splitext(basename)
                        random_name = uuid.uuid4().hex
                        dest_filepath = os.path.join(
                            os.path.dirname(dest_filepath), random_name + ext
                        )

                    os.rename(source_filepath, dest_filepath)

                    if (
                        count_files_in_directory(files_dest_folder)
                        >= MAX_FILES_FROM_ONE_REPO
                    ):
                        return
            except Exception as e:
                continue


GITHUB2TGLANG = {
    ### Other
    # 'JSON': 'TGLANG_LANGUAGE_JSON',
    ### Langsearch
    # "HTML": "TGLANG_LANGUAGE_HTML",
    # "Dockerfile": "TGLANG_LANGUAGE_DOCKER",
    # "TypeScript": "TGLANG_LANGUAGE_TYPESCRIPT",
    # "Cpp": "TGLANG_LANGUAGE_CPLUSPLUS",
    # "Python": "TGLANG_LANGUAGE_PYTHON",
    # "JavaScript": "TGLANG_LANGUAGE_JAVASCRIPT",
    # "C": "TGLANG_LANGUAGE_C",
    # "Csharp": "TGLANG_LANGUAGE_CSHARP",
    # "Java": "TGLANG_LANGUAGE_JAVA",
    # "CSS": "TGLANG_LANGUAGE_CSS",
    # "Go": "TGLANG_LANGUAGE_GO",
    # "SQL": "TGLANG_LANGUAGE_SQL",
    # "XML": "TGLANG_LANGUAGE_XML",
    # "Rust": "TGLANG_LANGUAGE_RUST",
    # "Shell": "TGLANG_LANGUAGE_SHELL",
    # "NGINX": "TGLANG_LANGUAGE_NGINX",
    # "Objective-C": "TGLANG_LANGUAGE_OBJECTIVE_C",
    # "PHP": "TGLANG_LANGUAGE_PHP",
    # "PowerShell": "TGLANG_LANGUAGE_POWERSHELL",
    # "Dart": "TGLANG_LANGUAGE_DART",
    # "Kotlin": "TGLANG_LANGUAGE_KOTLIN",
    # "Lua": "TGLANG_LANGUAGE_LUA",
    # "Ruby": "TGLANG_LANGUAGE_RUBY",
    # "Swift": "TGLANG_LANGUAGE_SWIFT",
    # "Solidity": "TGLANG_LANGUAGE_SOLIDITY",
    ### QSearch
    #'FUNC contract': 'TGLANG_LANGUAGE_FUNC',
    # 'TL Type Language': 'TGLANG_LANGUAGE_TL',
}

#

LIMIT_REPOS = 3000
LIMIT_FILES = 1000000

for github_lang in GITHUB2TGLANG.keys():
    # print(f"Processing language: {github_lang}")

    EXTENSIONS = TGLANG_LANGUAGE_EXTENSIONS[GITHUB2TGLANG[github_lang]]

    # repos = get_repositories_for_language(github_lang, LIMIT_REPOS)
    repos = get_repositories_for_query(github_lang, LIMIT_REPOS)

    file_progress = tqdm.tqdm(
        total=LIMIT_FILES, desc=f"Files ({github_lang})", position=0, leave=True
    )

    for repo in tqdm.tqdm(repos, desc=f"Repos ({github_lang})", leave=True):
        clone_and_process_repo(repo, github_lang, EXTENSIONS)

        files_path = os.path.join(SAVE_DIR, github_lang)
        num_files = count_files_in_directory(files_path)
        file_progress.update(num_files - file_progress.n)
        if num_files >= LIMIT_FILES:
            break

print("Done!")
