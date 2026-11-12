import os

from huggingface_hub import create_repo, upload_file

REPO = "goncayldz/clinc150-classical-models"
create_repo(REPO, exist_ok=True)

for f in ["models/tfidf_lr.joblib", "models/ffn.pt"]:
    upload_file(path_or_fileobj=f, path_in_repo=os.path.basename(f), repo_id=REPO)
    print("uploaded:", f)

print("https://huggingface.co/" + REPO)