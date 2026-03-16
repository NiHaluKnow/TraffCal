from pathlib import Path
import shutil
import kagglehub

dataset_id = "annnnguyen/metr-la-dataset"
download_path = Path(kagglehub.dataset_download(dataset_id))
target_path = Path("/home/nihal/Desktop/ML_50/Trafficformer/data/metr-la")

target_path.mkdir(parents=True, exist_ok=True)

for item in download_path.iterdir():
	destination = target_path / item.name
	if item.is_dir():
		shutil.copytree(item, destination, dirs_exist_ok=True)
	else:
		shutil.copy2(item, destination)

print(f"Downloaded from: {download_path}")
print(f"Saved to: {target_path}")