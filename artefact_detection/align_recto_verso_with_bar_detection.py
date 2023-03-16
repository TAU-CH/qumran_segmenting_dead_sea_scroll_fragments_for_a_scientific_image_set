import os
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import numpy as np
import orjson
from pathlib import Path
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from artefact_extract_tools import convert_numpy_to_memmap, decompose_multi_polygon, get_image_artefacts, get_corresponding_points, DetectExcludes

import logging
logging.basicConfig(
    filename=f"{Path(__file__).stem}.log",
    encoding="utf-8",
    level=logging.DEBUG,
    filemode="w",
)

# define output directories
RULERS_OUT_PATH = Path("./ruler_positions")
RECTO_VERSO_OUT_PATH = Path("./recto_verso_alignments")

def sort_matches(kp1, kp2, matches, accuracy, scale):

    good = [] # good matches

    # Loops through the matches and adds the ones that are considered good
    for m, n in matches:
        if m.distance < accuracy * n.distance:
            good.append(m)

    min_match_count = 1 # minimum number of good matches required
    
    # If the number of good matches is less than the minimum threshold, returns None
    if len(good) < min_match_count:
        print('min_match_count')
        return None

    # Reshapes the kepoint values into an array for further processing
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # Scales the keypoints and returns the good matches as a tuple
    return src_pts * scale, dst_pts * scale, good


def match_keypoints(des1, des2, kp1, kp2, scale, accuracy):
    
    # Check if keypoint and descriptor lists are not empty
    if (des1 is None and des2 is None) or (
            (des1 is None or len(des1) == 0)
            or (des2 is None or len(des2) == 0)
            or (kp1 is None or len(kp1) == 0)
            or (kp2 is None or len(kp2) == 0)
    ):
        print('match_keypoints returns none')
        return None

    # Create BFMatcher object
    match = cv2.BFMatcher()
    # Match descriptors from both images using knnMatch with k=2
    matches = match.knnMatch(des1, des2, k=2)
    # Sort matches by their distance and ratio test
    return sort_matches(kp1, kp2, matches, accuracy, scale)

def get_keypoints(analyzer, img, mask, scale):

    # Create a keypoint detector and descriptor extractor object
    kp_detect = analyzer()

    # Resize the image and mask if the scaling factor is not 1
    if scale != 1:
        img = cv2.resize(img, (int(img.shape[1] / scale), int(img.shape[0] / scale)))
        mask = cv2.resize(mask, (int(mask.shape[1] / scale), int(mask.shape[0] / scale)))

    # Convert the mask to a binary mask
    mask = np.where(mask >= 170, 255, 0).astype('uint8')

    # Detect and compute the keypoints and descriptors of the image
    return kp_detect.detectAndCompute(img, mask=mask)

def get_corresponding_points(img1, img1_mask, img2, img2_mask, scale, extractor): 

    # find the key points and descriptors with the specified extractor algorithm
    kp1, des1 = get_keypoints(extractor, img1, img1_mask, scale)
    kp2, des2 = get_keypoints(extractor, img2, img2_mask, scale)
    
    # set the threshold for matching accuracy
    accuracy = 0.8 
    
    # check if match_keypoints function returns None
    if match_keypoints(des1, des2, kp1, kp2, scale, accuracy)==None:
        print('match_keypoints is none')
        return
    
    # get the corresponding key points and descriptors using the match_keypoints function
    src, dst, good = match_keypoints(des1, des2, kp1, kp2, scale, accuracy)
    
    # return the corresponding key points, descriptors, and matches
    return src, dst, kp1, des1, kp2, des2, good

# This function returns the transformation required to register verso and recto images,
# applies it to the verso image, and saves the transformed image.
def get_transform_from_verso_to_recto(recto_ir, verso_ir, recto_color, verso_color, excludes, fragment_name: str, extractor, threshold):
    function = extractor
    scales = [4] #[2, 4, 8, 1, 16]

    # Loops through the scales and tries to find a transformation for each scale using the specified extractor function.
    for scale in scales:
        logging.debug(f"Trying function {function} and scale {scale}.")
        print(f"Trying function {function} and scale {scale}.")
        trans = calculate_transform_from_verso_to_recto(recto_ir, verso_ir, recto_color, verso_color, excludes, scale, extractor, threshold)
        
        # If no transformation is found for the given scale, logs a warning and continues to the next iteration.
        if not trans:
            logging.warning(f"Function {function} and scale {scale} found no registration")
            print(f"Function {function} and scale {scale} found no registration")
            continue

        # Retrieves the required information from the transformation dictionary and saves the transformed verso image.
        reverse_transform, verso_transform, recto_ruler_mask, verso_ruler_mask, num_inliers = trans
        
        # Save the transformed verso image
        masked_verso_ir = np.where(verso_ruler_mask == 255, verso_ir, 0)
        aligned_flipped_masked_verso_ir = cv2.warpAffine(np.flip(masked_verso_ir, axis=1), verso_transform, (recto_ir.shape[1], recto_ir.shape[0]))
        extractor_threshold_folder = Path("visual_results") / f"{extractor}_{threshold}"
        if not extractor_threshold_folder.exists():
            extractor_threshold_folder.mkdir(parents=True)
        transformed_verso_file = extractor_threshold_folder / f"aligned_verso_{fragment_name}.jpg"
        cv2.imwrite(str(transformed_verso_file), aligned_flipped_masked_verso_ir)
    
    # If no transformation was found for any scale, returns None.
    if not trans:
        return
    return reverse_transform, verso_transform, recto_ruler_mask, verso_ruler_mask, num_inliers

def calculate_transform_from_verso_to_recto(recto, verso, recto_color, verso_color, excludes, scale, extractor, threshold):
    
    # Create edge masks for recto and verso images
    recto_edge_mask = np.zeros((recto.shape[0], recto.shape[1]), dtype=np.uint8)
    recto_edge_mask[1000 : recto.shape[0] - 1000, 1000 : recto.shape[1] - 1000] = 255
    verso_edge_mask = np.zeros((verso.shape[0], verso.shape[1]), dtype=np.uint8)
    verso_edge_mask[1000 : verso.shape[0] - 1000, 1000 : verso.shape[1] - 1000] = 255

    # Read masks for recto and verso images
    recto_mask=cv2.imread('ruler_positions/'+excludes+'_recto_color.jpg',0)
    _, recto_mask = cv2.threshold(recto_mask, 250, 255, cv2.THRESH_BINARY_INV)

    verso_mask=cv2.imread('ruler_positions/'+excludes+'_verso_color.jpg',0)
    _, verso_mask = cv2.threshold(verso_mask, 250, 255, cv2.THRESH_BINARY_INV)

    # Create full masks for recto and verso images
    full_recto_mask = convert_numpy_to_memmap(
        np.logical_and(recto_edge_mask, recto_mask) * np.uint8(255),
        Path("./tmp/init_recto_mask.np"),
    )
    full_verso_mask = convert_numpy_to_memmap(
        np.logical_and(verso_edge_mask, verso_mask) * np.uint8(255),
        Path("./tmp/init_verso_mask.np"),
    )
    del recto_edge_mask, verso_edge_mask
   
    corresponding_points = [], []
    src = np.array([])
    dst = np.array([])
    # Get corresponding points of recto and verso images
    corresponding_points = get_corresponding_points(
        recto,
        full_recto_mask,
        np.flip(verso, axis=1),
        np.flip(full_verso_mask, axis=1),
        scale,
        extractor
    )
    
    # If corresponding points are not found, return None
    if corresponding_points is None:
        print('corresponding_points is none')
        return None

    # Get source and destination points, keypoints and descriptors
    src_pts, dst_pts, kp1, des1, kp2, des2, good = corresponding_points

    # If corresponding points are less than 3, return None
    if src_pts.size < 3 or dst_pts.size < 3:
        print('corrresponding_points are less than 3')
        return None
    
    # Estimate affine transform using RANSAC algorithm
    transform, mask = cv2.estimateAffinePartial2D(dst_pts, src_pts, method=cv2.RANSAC, ransacReprojThreshold=threshold)
    num_inliers = np.count_nonzero(mask)
    
    # Resize recto and verso images
    recto = cv2.resize(recto, (int(recto.shape[1] / scale), int(recto.shape[0] / scale)))
    verso = cv2.resize(verso, (int(verso.shape[1] / scale), int(verso.shape[0] / scale)))
    fliped_verso = np.flip(verso, axis=1)
    
    # Get inlier keypoints from the keypoints arrays
    inlier_keypoints1 = [kp1[m.queryIdx] for m, inlier in zip(good, mask) if inlier]
    inlier_keypoints2 = [kp2[m.trainIdx] for m, inlier in zip(good, mask) if inlier]
    
    # Draw inlier keypoints on the source and destination images
    # src_inliers1 = cv2.drawKeypoints(recto, inlier_keypoints1, mask, (0, 0, 255), 2)
    # dst_inliers1 = cv2.drawKeypoints(fliped_verso, inlier_keypoints2, mask, (0, 0, 255), 2)
    # cv2.imwrite("matchessrc1.png", src_inliers1)
    # cv2.imwrite("matchesdst1.png", dst_inliers1)

    # Draw matches between recto and verso images
    # matches1 = cv2.drawMatches(recto, kp1, fliped_verso, kp2, good, mask, flags=2)
    # cv2.imwrite("matches1.png", matches1)

    # Get inlier matches
    #inlier_matches = [m for i, m in enumerate(good) if mask[i]]
    
    # Draw matches between recto and verso images with inlier keypoints
    # matchesi = cv2.drawMatches(recto, inlier_keypoints1, fliped_verso, inlier_keypoints2, inlier_matches[:-1], None, flags=2)
    # cv2.imwrite("matchesi.png", matchesi)

    # Estimate reverse transform from destination to source points
    reverse_transform, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=threshold)

    # Return reverse and forward transforms, recto and verso masks and number of inliers
    return reverse_transform, transform, recto_mask, verso_mask, num_inliers

def align_recto_verso(fragment_name: str, recto_color_file: Path, recto_ir_file: Path, verso_color_file: Path, verso_ir_file: Path, extractor, threshold):
    
    excludes = recto_color_file.parts[2]
    
    # Read recto and verso images
    recto_color = cv2.imread(str(recto_color_file.absolute()))
    recto_ir = cv2.imread(str(recto_ir_file.absolute()), cv2.IMREAD_GRAYSCALE)
    verso_color = cv2.imread(str(verso_color_file.absolute()))
    verso_ir = cv2.imread(str(verso_ir_file.absolute()), cv2.IMREAD_GRAYSCALE)
    
    if get_transform_from_verso_to_recto(recto_ir, verso_ir, recto_color, verso_color, excludes, fragment_name, extractor, threshold)==None:
        return 0
        
    # Get reverse and verso transforms, recto and verso masks and number of inliers from the recto and verso images
    reverse_transform, verso_transform, recto_ruler_mask, verso_ruler_mask, num_inliers = get_transform_from_verso_to_recto(recto_ir, verso_ir, recto_color, verso_color, excludes, fragment_name, extractor, threshold)
    
    # Create output directories if they don't exist
    if not RULERS_OUT_PATH.exists():
        RULERS_OUT_PATH.mkdir()
    if not RECTO_VERSO_OUT_PATH.exists():
        RECTO_VERSO_OUT_PATH.mkdir()
    (RECTO_VERSO_OUT_PATH / f"""{fragment_name}_recto.json""").write_bytes(orjson.dumps(reverse_transform.tolist()))
    (RECTO_VERSO_OUT_PATH / f"""{fragment_name}_verso.json""").write_bytes(orjson.dumps(verso_transform.tolist())) 
  
    return num_inliers
    
if __name__ == "__main__":
    '''
    align_recto_verso(
        "1108_9",
        Path("test/images/1108_9/recto_color.jpg"),
        Path("test/images/1108_9/recto_infrared.jpg"),
        Path("test/images/1108_9/verso_color.jpg"),
        Path("test/images/1108_9/verso_infrared.jpg"))
    '''


# This function reads images from a given folder and its subfolders
# It returns a generator that yields the file paths of the images
def read_images_from_folder(folder_path):
    # Get the subfolders under the given folder path
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
    
    # Iterate through each subfolder
    for subfolder in subfolders:
        # Extract the name of the fragment from the subfolder path
        fragment_name = subfolder.split("/")[-1]
        
        # Get the file paths of the recto_color, recto_infrared, verso_color, and verso_infrared images
        recto_color_file = Path(subfolder) / "recto_color.jpg"
        recto_ir_file = Path(subfolder) / "recto_infrared.jpg"
        verso_color_file = Path(subfolder) / "verso_color.jpg"
        verso_ir_file = Path(subfolder) / "verso_infrared.jpg"
        
        # Yield the fragment name and file paths of the four images as a tuple
        yield fragment_name, recto_color_file, recto_ir_file, verso_color_file, verso_ir_file

# This function returns a dictionary that maps extractor types to thresholds to the number of inliers obtained
def combine_inliers_from_alignments(folder_path):
    # Define the extractors and thresholds to use
    extractors = [ cv2.KAZE_create, cv2.SIFT_create, cv2.AKAZE_create, cv2.ORB_create]
    thresholds = [21,19,17,15,13,11,9,7,5]
    
    # Create an empty dictionary to store the inliers
    inliers = {extractor: {threshold: [] for threshold in thresholds} for extractor in extractors}
    
    # Process each image from the folder
    for fragment_name, recto_color_file, recto_ir_file, verso_color_file, verso_ir_file in read_images_from_folder(folder_path):
        print("fragment name: ")
        print(fragment_name)
        # Iterate through each extractor and threshold combination
        for extractor in extractors:
            print("extractor: ")
            print(extractor)
            for threshold in thresholds:
                print("threshold: ")
                print(threshold)
                # Extract the number of inliers using the current extractor and threshold combination
                number_of_inliers = align_recto_verso(fragment_name, recto_color_file, recto_ir_file, verso_color_file, verso_ir_file, extractor, threshold)
                
                # Store the number of inliers for the current extractor and threshold combination
                inliers[extractor][threshold].append(number_of_inliers)
                
    # Return the dictionary of inliers
    return inliers


folder_path = "test/images"
inliers = combine_inliers_from_alignments(folder_path)

with open("inliers.txt", "w") as f:
    for extractor in inliers:
        for threshold in inliers[extractor]:
            f.write(str(extractor) + " " + str(threshold) + " " + str(sum(inliers[extractor][threshold])/len(inliers[extractor][threshold])) + "\n")


# read inliers.txt into a dictionary
inliers = {}
with open("inliers.txt") as f:
    lines = f.readlines()
    for line in lines:
        _, _, e, threshold, mean_inlier = line.strip().split()
        extractor = e.split('_')[0]
        threshold = int(threshold)
        mean_inlier = float(mean_inlier)
        if extractor not in inliers:
            inliers[extractor] = {}
        inliers[extractor][threshold] = mean_inlier

# plot the data
for extractor, thresholds_and_inliers in inliers.items():
    x = list(thresholds_and_inliers.keys())
    y = list(thresholds_and_inliers.values())
    plt.plot(x, y, label=extractor)

# add labels and legend
plt.xlabel("Threshold")
plt.ylabel("Mean number of inliers")
plt.legend()

# save the plot
plt.savefig("inliers.png")