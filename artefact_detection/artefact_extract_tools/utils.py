from collections import defaultdict
import cv2
import logging
import numpy as np
from pathlib import Path
from .rust_repair_polygon import poly_repair
from shapely import wkt
from shapely.geometry import Polygon

logger = logging.getLogger(__name__)


def convert_numpy_to_memmap(arr: np.ndarray, filename: Path):
    fp = np.memmap(filename, dtype=arr.dtype, mode='w+', shape=arr.shape)
    fp[:] = arr[:]
    return fp

def decompose_multi_polygon(multipolygon):
    polygons = []
    if hasattr(multipolygon, "geom_type") and multipolygon.geom_type == "Polygon":
        return [multipolygon]

    if hasattr(multipolygon, "geom_type") and multipolygon.geom_type == "MultiPolygon":
        for polygon in multipolygon.geoms:
            polygons = polygons + decompose_multi_polygon(polygon)
        return polygons
        
    for polygon in multipolygon:
        polygons = polygons + decompose_multi_polygon(polygon)

    return polygons

def shapes_from_contours(contours, hierarchy, min_area=10):
    """See https://stackoverflow.com/questions/60971260/how-to-transform-contours-obtained-from-opencv-to-shp-file-polygons"""
    if not contours:
        return []
    # now messy stuff to associate parent and child contours
    cnt_children = defaultdict(list)
    child_contours = set()
    logging.debug("Mapping contour heirarchies")
    assert hierarchy.shape[0] == 1
    # http://docs.opencv.org/3.1.0/d9/d8b/tutorial_py_contours_hierarchy.html
    for idx, (_, _, _, parent_idx) in enumerate(hierarchy[0]):
        if parent_idx != -1:
            child_contours.add(idx)
            cnt_children[parent_idx].append(contours[idx])
    # create actual polygons filtering by area (removes artifacts)
    all_polygons = []
    logging.debug(f"There are {len(contours)} contours to process")
    for idx, cnt in enumerate(contours):
        if idx not in child_contours and cv2.contourArea(cnt) >= min_area:
            assert cnt.shape[1] == 1
            poly = Polygon(
                shell=cnt[:, 0, :],
                holes=[
                    c[:, 0, :]
                    for c in cnt_children.get(idx, [])
                    if cv2.contourArea(c) >= min_area
                ],
            )
            logging.debug(poly.wkt)
            if not poly.is_valid:
                poly = fix_polygon(poly)

            if poly.area > 0:
                all_polygons.append(poly.simplify(0.05, preserve_topology=False))

    logging.debug("Parsed all contours into polygons")
    return all_polygons


def get_image_artefacts(img):
    final_contours, final_hierarchy = cv2.findContours(
        img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
    )
    return shapes_from_contours(final_contours, final_hierarchy)

def fix_polygon(poly: Polygon):
    if poly.is_valid:
        return poly

    logging.debug("Trying the buffered polygon trick")
    buffered_poly = poly.buffer(0)
    if buffered_poly.is_valid and buffered_poly.area > 0:
        return buffered_poly

    logging.debug("Trying my rust polygon repair suite")
    try:
        repaired_poly = poly_repair(poly.wkt)
        if repaired_poly == "INVALIDGEOMETRY":
            return Polygon()

        return wkt.loads(repaired_poly)
    except:
        return Polygon()


def flip_transform_verso(verso_image, recto_image, transform):
    return cv2.warpAffine(
        np.flip(verso_image, axis=1), transform, (recto_image.shape[1], recto_image.shape[0])
    )