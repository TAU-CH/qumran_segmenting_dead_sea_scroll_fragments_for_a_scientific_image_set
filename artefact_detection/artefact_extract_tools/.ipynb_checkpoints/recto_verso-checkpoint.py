import cv2
import logging
import numpy as np
from numba import jit
from pathlib import Path
import logging

from artefact_extract_tools import convert_numpy_to_memmap

logger = logging.getLogger(__name__)
numba_logger = logging.getLogger("numba")
numba_logger.setLevel(logging.WARNING)


class DetectExcludes:
    def __init__(self, excludes: list[Path], scale=20):
        self.__scale = scale
        # self.__exclude_rotations = [0, 5, 85, 90, 95, 175, 180, 185, 265, 270, 275, 355]
        # self.__exclude_rotations = [0, 90, 180, 270]
        self.__excludes = [self.__prepare_exclude_image(x) for x in excludes]

    def __prepare_exclude_image(self, img_path: Path) -> list[np.ndarray]:
        img = self.__prepare_image(cv2.imread(str(img_path)))
        img_90 = np.rot90(img)
        img_180 = np.rot90(img, 2)
        img_270 = np.rot90(img, 3)
        return [img, img_90, img_180, img_270]

    def __prepare_image(self, img: np.ndarray) -> np.ndarray:
        small_img = self.__resize_image(img)
        if len(img.shape) > 2:
            small_img = self.__get_lab_gray(small_img)
        return small_img

    def __resize_image(self, img: np.ndarray) -> np.ndarray:
        return cv2.resize(
            img, (int(img.shape[1] / self.__scale), int(img.shape[0] / self.__scale))
        )

    def __get_lab_gray(self, img):
        def get_hue(red: int, green: int, blue: int):
            min_val = min(min(red, green), blue)
            max_val = max(max(red, green), blue)

            if min_val == max_val:
                return 0

            hue = 0.0
            if max_val == red:
                hue = (green - blue) / (max_val - min_val)

            elif max == green:
                hue = 2.0 + (blue - red) / (max_val - min_val)

            else:
                hue = 4.0 + (red - green) / (max_val - min_val)

            hue = hue * 60
            if hue < 0:
                hue = hue + 360

            return int(hue)

        lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab_hue = np.zeros((lab_img.shape[0], lab_img.shape[1]))
        for y in range(lab_img.shape[0]):
            for x in range(lab_img.shape[1]):
                lab_hue[y][x] = get_hue(
                    int(lab_img[y][x][0]), int(lab_img[y][x][1]), int(lab_img[y][x][2])
                )
        return lab_hue.astype(np.uint8)

    def __detect_excludes_rect(self, img: np.ndarray) -> list[any]:
        img = img.copy()
        max_region = {
            "top": 0,
            "bottom": img.shape[0],
            "left": 0,
            "right": img.shape[1],
            "rotation_idx": 0,
        }
        matches = []
        for exclude in self.__excludes:
            lowest_score = 1
            best_image_region = {}
            for rotation_idx, rotated_exclude in enumerate(exclude):
                w, h = rotated_exclude.shape[::-1]
                if img.shape[0] < h or img.shape[1] < w:
                    continue

                # Apply template Matching
                res = cv2.matchTemplate(img, rotated_exclude, cv2.TM_CCOEFF_NORMED)
                _, _, _, max_loc = cv2.minMaxLoc(res)
                top_left = max_loc
                bottom_right = (top_left[0] + w, top_left[1] + h)
                similarity = cv2.matchShapes(
                    rotated_exclude,
                    img[top_left[1] : bottom_right[1], top_left[0] : bottom_right[0]],
                    cv2.CONTOURS_MATCH_I3,
                    0,
                )
                if similarity < lowest_score:
                    lowest_score = similarity
                    best_image_region = {
                        "top": top_left[1],
                        "bottom": bottom_right[1],
                        "left": top_left[0],
                        "right": bottom_right[0],
                        "rotation_idx": rotation_idx,
                    }
            # if lowest_score <= 0.01:
            #     img[
            #         best_image_region["top"] : best_image_region["bottom"],
            #         best_image_region["left"] : best_image_region["right"],
            #     ] = 50
            matches.append(max_region if lowest_score > 0.03 else best_image_region)

        return matches

    def __sort_matches(self, kp1, kp2, matches, accuracy):
        good = []
        for m, n in matches:
            if m.distance < accuracy * n.distance:
                good.append(m)

        min_match_count = 10
        if len(good) > min_match_count:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            return src_pts, dst_pts
        else:
            return None

    def __match_keypoints(self, des1, des2, kp1, kp2, accuracy):
        if (des1 is None and des2 is None) or (
            (des1 is None or len(des1) == 0)
            or (des2 is None or len(des2) == 0)
            or (kp1 is None or len(kp1) == 0)
            or (kp2 is None or len(kp2) == 0)
        ):
            return None
        # FLANN_INDEX_KDTREE = 0
        # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        # search_params = dict(checks=50)
        # match = cv2.FlannBasedMatcher(index_params, search_params)
        # matches = match.knnMatch(des1, des2, k=2)
        match = cv2.BFMatcher()
        matches = match.knnMatch(des1, des2, k=2)

        return self.__sort_matches(kp1, kp2, matches, accuracy)

    def __get_keypoints(self, analyzer, img, mask):
        kp_detect = analyzer()
        return kp_detect.detectAndCompute(img, mask=mask)

    def __get_corresponding_points(
        self, img1, img1_mask, img2, img2_mask, accuracy=0.5, algorithm=cv2.SIFT_create
    ):
        # find the key points and descriptors with user defined algorithm
        kp1, des1 = self.__get_keypoints(algorithm, img1, img1_mask)
        kp2, des2 = self.__get_keypoints(algorithm, img2, img2_mask)

        return self.__match_keypoints(des1, des2, kp1, kp2, accuracy)

    def __mask_element(self, image: np.ndarray, exclude_element: np.ndarray, region):
        image_region_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        width = abs(region["left"] - region["right"])
        height = abs(region["top"] - region["bottom"])
        top = max(int(region["top"] - (height / 2)), 0)
        bottom = min(int(region["bottom"] + (height / 2)), image.shape[0])
        left = max(int(region["left"] - (width / 2)), 0)
        right = min(int(region["right"] + (width / 2)), image.shape[1])
        image_region_mask[top:bottom, left:right] = 255
        loc = self.__get_corresponding_points(
            image,
            image_region_mask,
            exclude_element,
            np.full(
                (exclude_element.shape[0], exclude_element.shape[1]),
                255,
                dtype=np.uint8,
            ),
            accuracy=0.7,
            algorithm=cv2.SIFT_create,
        )
        if loc is not None:
            transform, _ = cv2.estimateAffinePartial2D(
                loc[1], loc[0], method=cv2.RANSAC, ransacReprojThreshold=10
            )
            blank_mask = np.full(
                (exclude_element.shape[0], exclude_element.shape[1]), 0, dtype=np.ubyte
            )
            trans_mask = cv2.warpAffine(
                blank_mask,
                transform,
                (image.shape[1], image.shape[0]),
                borderValue=(255, 255, 255),
            )

            image_region_mask = trans_mask

        return image_region_mask.astype(np.uint8)

    def detect_exclude_regions(self, img: np.ndarray, flip=False) -> np.ndarray:
        image_shape = img.shape
        prepared_img = self.__prepare_image(img)
        if flip:
            prepared_img = np.flip(prepared_img, axis=1)

        gross_exclude_regions = self.__detect_excludes_rect(prepared_img)
        excludes_mask = np.full(
            (prepared_img.shape[0], prepared_img.shape[1]), False, dtype=bool
        )

        for idx, exclude_region in enumerate(gross_exclude_regions):
            if exclude_region is None:
                continue

            exclude_element = self.__excludes[idx][exclude_region["rotation_idx"]]
            refined_mask = self.__mask_element(
                prepared_img,
                exclude_element,
                exclude_region,
            )
            excludes_mask = excludes_mask + refined_mask.astype(bool)
        excludes_mask = cv2.resize(
            excludes_mask.astype(np.uint8) * 255,
            (
                image_shape[1],
                image_shape[0],
            ),
        )
        return excludes_mask if not flip else np.flip(excludes_mask, axis=1)

def sort_matches(kp1, kp2, matches, accuracy, scale):
    good = []
    for m, n in matches:
        if m.distance < accuracy * n.distance:
            good.append(m)

    min_match_count = 10
    if len(good) < min_match_count:
        return None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    return src_pts * scale, dst_pts * scale


def match_keypoints(des1, des2, kp1, kp2, scale, accuracy):
    if (des1 is None and des2 is None) or (
        (des1 is None or len(des1) == 0)
        or (des2 is None or len(des2) == 0)
        or (kp1 is None or len(kp1) == 0)
        or (kp2 is None or len(kp2) == 0)
    ):
        return None
    # FLANN_INDEX_KDTREE = 0
    # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    # search_params = dict(checks=50)
    # match = cv2.FlannBasedMatcher(index_params, search_params)
    # matches = match.knnMatch(des1, des2, k=2)
    match = cv2.BFMatcher()
    matches = match.knnMatch(des1, des2, k=2)
    return sort_matches(kp1, kp2, matches, accuracy, scale)


def get_keypoints(analyzer, img, mask, scale):
    kp_detect = analyzer()
    if scale != 1:
        img = cv2.resize(img, (int(img.shape[1] / scale), int(img.shape[0] / scale)))
        mask = cv2.resize(mask, (int(img.shape[1] / scale), int(img.shape[0] / scale)))

    return kp_detect.detectAndCompute(img, mask=mask)


def get_corresponding_points(
    img1, img1_mask, img2, img2_mask, scale=1, accuracy=0.65, algorithm=cv2.SIFT_create
):
    # find the key points and descriptors with SIFT
    kp1, des1 = get_keypoints(algorithm, img1, img1_mask, scale)
    kp2, des2 = get_keypoints(algorithm, img2, img2_mask, scale)

    return match_keypoints(des1, des2, kp1, kp2, scale, accuracy)


def get_excludes_mask(img, inverse=False, scale=2):
    color_card = cv2.imread("./rulers/color-chart.tif")[..., ::-1]
    color_ruler = cv2.imread("./rulers/color-ruler.tif")[..., ::-1]
    if inverse:
        color_card = np.flip(color_card, axis=1)
        color_ruler = np.flip(color_ruler, axis=1)
        img = np.flip(img, axis=1)
    excludes = [color_ruler, color_card]

    return mask_elements(img, excludes, scale)


def get_transform_from_verso_to_recto(recto, verso, recto_color, verso_color, scale=2):
    recto_edge_mask = np.zeros((recto.shape[0], recto.shape[1]), dtype=np.uint8)
    recto_edge_mask[1000 : recto.shape[0] - 1000, 1000 : recto.shape[1] - 1000] = 255
    verso_edge_mask = np.zeros((verso.shape[0], verso.shape[1]), dtype=np.uint8)
    verso_edge_mask[1000 : verso.shape[0] - 1000, 1000 : verso.shape[1] - 1000] = 255
    recto_mask = convert_numpy_to_memmap(
        get_excludes_mask(recto_color, inverse=False, scale=scale),
        Path("./recto_rulers_mask.np"),
    )
    verso_mask = convert_numpy_to_memmap(
        get_excludes_mask(verso_color, inverse=True, scale=scale),
        Path("./verso_rulers_mask.np"),
    )
    full_recto_mask = convert_numpy_to_memmap(
        np.logical_and(recto_edge_mask, recto_mask) * np.uint8(255),
        Path("./init_recto_mask.np"),
    )
    full_verso_mask = convert_numpy_to_memmap(
        np.logical_and(verso_edge_mask, verso_mask) * np.uint8(255),
        Path("./init_verso_mask.np"),
    )
    del recto_edge_mask, verso_edge_mask

    functions = [cv2.AKAZE_create, cv2.SIFT_create, cv2.KAZE_create, cv2.ORB_create]
    corresponding_points = [], []
    src = np.array([])
    dst = np.array([])
    for function in functions:
        corresponding_points = get_corresponding_points(
            recto,
            full_recto_mask,
            np.flip(verso, axis=1),
            full_verso_mask,
            scale=scale,
            algorithm=function,
        )
        if corresponding_points is not None:
            src, dst = corresponding_points
            if src.size > 2 and dst.size > 2:
                break

    if src.size < 3 or dst.size < 3:
        return None

    reverse_transform = transform = None
    for i in reversed(range(1, 15, 2)):
        transform, _ = cv2.estimateAffinePartial2D(
            dst, src, method=cv2.RANSAC, ransacReprojThreshold=i
        )
        if transform is not None:
            break
    for i in reversed(range(1, 15, 2)):
        reverse_transform, _ = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=i
        )
        if reverse_transform is not None:
            break

    return reverse_transform, transform, recto_mask, verso_mask

def mask_elements(image, exclude_elements, scale=4):
    resized_image = cv2.resize(
        image, (int(image.shape[1] / scale), int(image.shape[0] / scale))
    )
    mask = np.full(
        (resized_image.shape[0], resized_image.shape[1]), 255, dtype=np.ubyte
    )
    for mask_image in exclude_elements:
        mask_image = cv2.resize(
            mask_image,
            (int(mask_image.shape[1] / scale), int(mask_image.shape[0] / scale)),
        )
        loc = get_corresponding_points(
            resized_image,
            np.full(
                (resized_image.shape[0], resized_image.shape[1]), 255, dtype=np.ubyte
            ),
            mask_image,
            np.full((mask_image.shape[0], mask_image.shape[1]), 255, dtype=np.ubyte),
            scale=scale,
            accuracy=0.9,
            algorithm=cv2.SIFT_create,
        )
        if loc is not None:
            transform, _ = cv2.estimateAffinePartial2D(
                loc[1], loc[0], method=cv2.RANSAC, ransacReprojThreshold=10
            )
            blank_mask = np.full(
                (mask_image.shape[0], mask_image.shape[1]), 0, dtype=np.ubyte
            )
            trans_mask = cv2.warpAffine(
                blank_mask,
                transform,
                (resized_image.shape[1], resized_image.shape[0]),
                borderValue=(255, 255, 255),
            )
            mask = _combine_masks(mask, trans_mask)
            del blank_mask
            del trans_mask
    return cv2.resize(mask, (image.shape[1], image.shape[0]))

@jit(nopython=True, parallel=True)
def _combine_masks(x, y):
    return np.where(x == 0, x, y)