import json
from pathlib import Path

import nltk
from nltk.corpus import wordnet

from .classes import IMAGENET2012_CLASSES


def class_name_to_synonyms(class_name: str) -> list:
    return class_name.strip().replace(", ", ",").replace(" ", "_").split(",")


def main() -> None:
    # nltk.download()
    categories = []
    output_json_path = Path("./imagenet_1k_wordnet_class_info.json")
    for i, key in enumerate(IMAGENET2012_CLASSES.keys()):
        class_name = IMAGENET2012_CLASSES[key]
        # from https://github.com/facebookresearch/Detic/blob/main/tools/get_imagenet_21k_full_tar_json.py#L42-L52
        synset = wordnet.synset_from_pos_and_offset("n", int(key[1:]))
        synonyms = [x.name() for x in synset.lemmas()]
        category = {
            "id": i + 1,
            "synset": synset.name(),
            "name": synonyms[0],
            "def": synset.definition(),
            "synonyms": synonyms,
        }
        is_same = synonyms == class_name_to_synonyms(class_name)
        if not is_same:
            print(f"{i}, {synonyms}, {class_name}")
        categories.append(category)

    data = {"categories": categories, "images": [], "annotations": []}
    print(f"save image net class info {str(output_json_path)}")
    with output_json_path.open("w") as f:
        json.dump(data, f)


if __name__ == "__main__":
    main()
