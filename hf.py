from pathlib import Path

from huggingface_hub import HfApi, upload_file, upload_folder

USER = "Abhinavexists"

ROOT = Path("/home/abhinav/Projects/SeeSharp")
api = HfApi()

teacher_repo = f"{USER}/SeeSharp"
api.create_repo(teacher_repo, repo_type="model", exist_ok=True)

upload_folder(
    repo_id=teacher_repo,
    folder_path=str(ROOT / "ersvr" / "models"),
    path_in_repo="ersvr/models",
    repo_type="model",
)

upload_file(
    repo_id=teacher_repo,
    path_or_fileobj=str(ROOT / "models" / "teacher_models" / "ersvr_best.pth"),
    path_in_repo="weights/ersvr_best.pth",
    repo_type="model",
)

student_repo = f"{USER}/SeeSharp"
api.create_repo(student_repo, repo_type="model", exist_ok=True)

upload_file(
    repo_id=student_repo,
    path_or_fileobj=str(ROOT / "ersvr" / "models" / "student.py"),
    path_in_repo="ersvr/models/student.py",
    repo_type="model",
)

upload_file(
    repo_id=student_repo,
    path_or_fileobj=str(ROOT / "models" / "student_models" / "student_best.pth"),
    path_in_repo="weights/student_best.pth",
    repo_type="model",
)

print("Done. Teacher:", teacher_repo, "| Student:", student_repo)
