"""Upload versioned project files to an existing authorized Hugging Face Docker Space."""
from __future__ import annotations
import argparse
from pathlib import Path
import subprocess

from huggingface_hub import CommitOperationAdd, HfApi, get_token

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="Nkreddy15/employee-attrition-api")
    args = parser.parse_args()
    if not get_token():
        parser.error("Sign in first using hf auth login (or supply HF_TOKEN through the environment).")
    api = HfApi()
    api.auth_check(repo_id=args.repo_id, repo_type="space", write=True)
    info = api.space_info(args.repo_id)
    if info.sdk != "docker":
        parser.error("The existing Space must use the Docker SDK.")
    paths = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    paths = [path for path in paths if path and not path.startswith((".github/", "screenshot/"))]
    operations = [CommitOperationAdd(path_in_repo=path, path_or_fileobj=str(ROOT / path)) for path in paths]
    result = api.create_commit(repo_id=args.repo_id, repo_type="space", operations=operations,
                               parent_commit=info.sha, commit_message="Deploy monitored attrition API and audited schema v2")
    print(result.commit_url)
    print("Build started; check /health on", info.host, "before claiming deployment success.")


if __name__ == "__main__":
    main()
