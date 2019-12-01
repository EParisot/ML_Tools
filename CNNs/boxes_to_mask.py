import click
import json


@click.command()
@click.option('-i', 'images_path', default="data", help="path to find images")
@click.option('-l', 'labels_file', default="data/bboxes.json", help="path to store labels")
def main(images_path, labels_file):
    with open(labels_file) as f:
        data = json.load(f)
    for img in data:
        for c in data[img]:
            for i, rect in enumerate(data[img][c]):
                [x, y, w, h] = rect
                converted = [[x - w / 2, y - h / 2], [x - w / 2, y + h / 2], [x + w / 2, y + h / 2], [x + w / 2, y - h / 2]]
                data[img][c][i] = converted
    with open("converted.json", mode="w") as f:
        json.dump(data, f)
            

if __name__ == "__main__":
    main()
