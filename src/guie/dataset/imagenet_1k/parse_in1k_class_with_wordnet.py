import logging

import pandas as pd
from nltk.corpus import wordnet as wn

from .classes import IMAGENET2012_CLASSES
from .constants import in_1k_const

logger = logging.getLogger(__name__)


def class_name_to_synonyms(class_name: str) -> list:
    return class_name.strip().replace(", ", ",").replace(" ", "_").split(",")


# can use ood label
NON_TARGET = [
    wn.synset("animal.n.01"),
    wn.synset("plant_organ.n.01"),
    wn.synset("vascular_plant.n.01"),
    wn.synset("plant_part.n.01"),
    wn.synset("plant.n.02"),
    wn.synset("traffic_light.n.01"),
    wn.synset("obstruction.n.01"),
    wn.synset("weapon.n.01"),
    wn.synset("military_vehicle.n.01"),
    wn.synset("aircraft.n.01"),
    wn.synset("balloon.n.01"),
]

# had better to ignore
# Synset('geological_formation.n.01')
# S for toy

# TARGET = [
#     Synset('fish.n.01')
#     Synset('decapod_crustacean.n.01')  # crab robuster
#     ]
TARGET_DOMAIN = [
    wn.synset("clothing.n.01"),
    wn.synset("food.n.01"),
    wn.synset("food.n.02"),  # ingredients
    wn.synset("musical_instrument.n.01"),
    wn.synset("magazine.n.01"),
    wn.synset("clock.n.01"),
    wn.synset("bag.n.01"),
    wn.synset("furniture.n.01"),
    wn.synset("dressing.n.04"),
    wn.synset("game.n.09"),
    wn.synset("pen.n.01"),
    wn.synset("ball.n.01"),
    wn.synset("commodity.n.01"),
    wn.synset("fabric.n.01"),
    wn.synset("binoculars.n.01"),
    wn.synset("outbuilding.n.01"),
    wn.synset("cleaning_implement.n.01"),
    wn.synset("body_armor.n.01"),
    wn.synset("building.n.01"),
    wn.synset("bell_cote.n.01"),  #
    wn.synset("bread.n.01"),
    wn.synset("boot.n.01"),
    wn.synset("roof.n.01"),
    wn.synset("footwear.n.02"),
    wn.synset("jewelry.n.01"),
    wn.synset("pillow.n.01"),
    wn.synset("armor_plate.n.01"),
    wn.synset("rug.n.01"),
    wn.synset("bedclothes.n.01"),
    wn.synset("face_mask.n.01"),
    wn.synset("book_jacket.n.01"),
    wn.synset("fruit.n.01"),
]
NEED_IGNORED = [
    wn.synset("sports_equipment.n.01"),
    wn.synset("container.n.01"),
    wn.synset("barrel.n.02"),  #
    wn.synset("barrow.n.03"),  #
    wn.synset("explorer.n.01"),  #
    wn.synset("contestant.n.01"),  #  player
    wn.synset("instrumentality.n.03"),
    wn.synset("geological_formation.n.01"),
]
GLOBAL_DOMAIN = [
    wn.synset("airplane.n.01"),
    wn.synset("car.n.01"),
    wn.synset("shop.n.01"),
    wn.synset("bobsled.n.02"),
]
MULTI_FILTERING = [
    [wn.synset("artifact.n.01"), wn.synset("structure.n.01")],
]


WRONG_CLASS = [
    "custard apple",
    "banana",
    "pineapple",
    "hermit crab",
]


def find_ood_class_from_in1k() -> pd.DataFrame:
    # nltk.download()
    categories = []
    for label_id, key in enumerate(IMAGENET2012_CLASSES.keys()):
        class_name = IMAGENET2012_CLASSES[key]
        synset = wn.synset_from_pos_and_offset("n", int(key[1:]))
        synonyms = [x.name() for x in synset.lemmas()]

        hypernym_paths = set(synset.hypernym_paths()[0])
        need_ignored = hypernym_paths & set(NEED_IGNORED)
        non_target_check = hypernym_paths & set(NON_TARGET)
        target_check = hypernym_paths & set(TARGET_DOMAIN)
        global_check = hypernym_paths & set(GLOBAL_DOMAIN)
        multi_checks = []
        for multi_domain in MULTI_FILTERING:
            multi_check = hypernym_paths & set(multi_domain)
            if len(multi_check) == len(multi_domain):
                multi_checks.append(multi_check)

        is_out_of_domain = False
        if len(need_ignored) > 0:
            logger.debug(f"need_ignored : {need_ignored}")
        elif len(target_check) > 0:
            logger.debug(f"target: {target_check}")
        elif len(multi_checks) > 0:
            logger.debug(f"multi_check: {multi_checks}")
        elif len(non_target_check) > 0:
            logger.debug(f"non target: {non_target_check}")
            is_out_of_domain = True
        elif len(global_check) > 0:
            logger.debug(f"global: {global_check}")
            is_out_of_domain = True
        else:
            logger.debug("here")
            logger.debug(f"{label_id}, {synonyms}")
            logger.debug(f"{hypernym_paths}")
            pass

        category = {
            in_1k_const.LABEL: label_id,  # start from zero for label id
            "synset": synset.name(),
            "name": synonyms[0],
            "def": synset.definition(),
            "synonyms": synonyms,
            in_1k_const.IS_OOD: is_out_of_domain,
        }
        is_same = synonyms == class_name_to_synonyms(class_name)
        if not is_same:
            logger.debug(f"{label_id}, {synonyms}, {class_name}")

        if class_name in WRONG_CLASS:
            logger.info("here")
            logger.info(f"{label_id}, {synonyms}")
            logger.info(f"{hypernym_paths}")

        categories.append(category)

    df = pd.DataFrame(categories)
    logger.debug(df.head())
    logger.debug(df["is_out_of_domain"].value_counts())
    return df
