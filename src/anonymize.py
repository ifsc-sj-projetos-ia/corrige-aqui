import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def blur_faces(image, model, conf):
    prediction = model.predict(image, conf=conf, verbose=False)[0]
    anonymized = image.copy()
    boxes = prediction.boxes.xyxy.cpu().numpy().astype(int)
    for x1, y1, x2, y2 in boxes:
        region = anonymized[y1:y2, x1:x2]
        if region.shape[0] >= 2 and region.shape[1] >= 2:
            k = max(3, min(region.shape[0], region.shape[1]) // 2 * 2 + 1)
            anonymized[y1:y2, x1:x2] = cv2.GaussianBlur(region, (k, k), 0)
    return anonymized, len(boxes)


def collect_images(input_path):
    if input_path.is_dir():
        return sorted(p for p in input_path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    return [input_path]


def main():
    parser = argparse.ArgumentParser(
        description="Detecta rostos com o modelo treinado e os borra com desfoque gaussiano (demo de anonimizacao)."
    )
    parser.add_argument("entrada", type=Path, help="Imagem ou pasta com imagens")
    parser.add_argument("--weights", type=Path, default=Path("models/best.pt"))
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--out-dir", type=Path, default=Path("results/anonimizadas"))
    args = parser.parse_args()

    if not args.entrada.exists():
        parser.error(f"entrada nao encontrada: {args.entrada}")
    if not args.weights.exists():
        parser.error(f"pesos nao encontrados: {args.weights}")

    images = collect_images(args.entrada)
    if not images:
        parser.error(f"nenhuma imagem encontrada em {args.entrada}")

    model = YOLO(str(args.weights))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for path in images:
        image = cv2.imread(str(path))
        if image is None:
            print(f"{path.name}: nao foi possivel ler a imagem, pulando")
            continue
        anonymized, n_faces = blur_faces(image, model, args.conf)
        out_path = args.out_dir / f"{path.stem}_anonimizada{path.suffix}"
        cv2.imwrite(str(out_path), anonymized)
        print(f"{path.name}: {n_faces} rosto(s) borrado(s) -> {out_path}")


if __name__ == "__main__":
    main()
