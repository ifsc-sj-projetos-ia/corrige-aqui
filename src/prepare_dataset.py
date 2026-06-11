import argparse
import random
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

import cv2

BASE_URL = "https://huggingface.co/datasets/CUHK-CSE/wider_face/resolve/main/data"

ARCHIVES = {
    "WIDER_train.zip": f"{BASE_URL}/WIDER_train.zip",
    "WIDER_val.zip": f"{BASE_URL}/WIDER_val.zip",
    "wider_face_split.zip": f"{BASE_URL}/wider_face_split.zip",
}


def report_progress(block_num, block_size, total_size):
    if total_size > 0:
        percent = min(100, block_num * block_size * 100 // total_size)
        sys.stdout.write(f"\r  {percent}%")
        sys.stdout.flush()


def download_and_extract(raw_dir):
    raw_dir.mkdir(parents=True, exist_ok=True)
    for filename, url in ARCHIVES.items():
        archive_path = raw_dir / filename
        if archive_path.exists():
            print(f"{filename} ja existe, pulando download")
        else:
            print(f"Baixando {filename}...")
            urllib.request.urlretrieve(url, archive_path, report_progress)
            print()
        marker = raw_dir / filename.replace(".zip", "")
        if not marker.exists():
            print(f"Extraindo {filename}...")
            with zipfile.ZipFile(archive_path) as zf:
                zf.extractall(raw_dir)


def parse_annotations(annotation_file, max_faces):
    entries = {}
    lines = annotation_file.read_text().splitlines()
    i = 0
    while i < len(lines):
        name = lines[i].strip()
        count = int(lines[i + 1])
        num_lines = max(count, 1)
        boxes = []
        for j in range(num_lines):
            parts = lines[i + 2 + j].split()
            x, y, w, h = (int(p) for p in parts[:4])
            invalid = int(parts[7])
            if w > 0 and h > 0 and invalid == 0:
                boxes.append((x, y, w, h))
        if boxes and len(boxes) <= max_faces:
            entries[name] = boxes
        i += 2 + num_lines
    return entries


def to_yolo_label(boxes, img_width, img_height):
    lines = []
    for x, y, w, h in boxes:
        cx = min(max((x + w / 2) / img_width, 0.0), 1.0)
        cy = min(max((y + h / 2) / img_height, 0.0), 1.0)
        nw = min(w / img_width, 1.0)
        nh = min(h / img_height, 1.0)
        lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    return "\n".join(lines) + "\n"


def build_split(names, entries, images_dir, split_dir):
    out_images = split_dir / "images"
    out_labels = split_dir / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)
    converted = 0
    for name in names:
        src = images_dir / name
        img = cv2.imread(str(src))
        if img is None:
            continue
        height, width = img.shape[:2]
        flat_name = name.replace("/", "_")
        shutil.copyfile(src, out_images / flat_name)
        label_path = out_labels / (Path(flat_name).stem + ".txt")
        label_path.write_text(to_yolo_label(entries[name], width, height))
        converted += 1
    return converted


def write_dataset_yaml(subset_dir):
    content = (
        f"path: {subset_dir.resolve().as_posix()}\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n"
        "names:\n"
        "  0: face\n"
    )
    (subset_dir / "dataset.yaml").write_text(content)


def main():
    parser = argparse.ArgumentParser(
        description="Baixa o WIDER Face, converte as anotacoes para o formato YOLO e gera um subset com splits de treino, validacao e teste."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--train-size", type=int, default=1500)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--test-size", type=int, default=300)
    parser.add_argument("--max-faces", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    raw_dir = args.data_dir / "raw"
    subset_dir = args.data_dir / "wider_subset"

    if not args.skip_download:
        download_and_extract(raw_dir)

    split_dir = raw_dir / "wider_face_split"
    train_entries = parse_annotations(split_dir / "wider_face_train_bbx_gt.txt", args.max_faces)
    val_entries = parse_annotations(split_dir / "wider_face_val_bbx_gt.txt", args.max_faces)
    print(f"Imagens com anotacoes validas (ate {args.max_faces} faces): {len(train_entries)} (treino), {len(val_entries)} (validacao)")

    rng = random.Random(args.seed)
    train_pool = rng.sample(sorted(train_entries), min(args.train_size, len(train_entries)))
    test_pool = rng.sample(sorted(val_entries), min(args.test_size, len(val_entries)))

    val_count = int(len(train_pool) * args.val_fraction)
    val_names = train_pool[:val_count]
    train_names = train_pool[val_count:]

    if subset_dir.exists():
        shutil.rmtree(subset_dir)

    train_images_dir = raw_dir / "WIDER_train" / "images"
    val_images_dir = raw_dir / "WIDER_val" / "images"

    n_train = build_split(train_names, train_entries, train_images_dir, subset_dir / "train")
    n_val = build_split(val_names, train_entries, train_images_dir, subset_dir / "val")
    n_test = build_split(test_pool, val_entries, val_images_dir, subset_dir / "test")

    write_dataset_yaml(subset_dir)

    print(f"Subset gerado em {subset_dir.resolve()}")
    print(f"  treino: {n_train} imagens")
    print(f"  validacao: {n_val} imagens")
    print(f"  teste: {n_test} imagens")
    print(f"Configuracao: {(subset_dir / 'dataset.yaml').resolve()}")


if __name__ == "__main__":
    main()
