import cv2
import numpy as np
import glob
import os
from tqdm import tqdm


# Extracts and matches keypoints between two images using SIFT and FLANN.
def extract_and_match_keypoints(image1, image2, max_keypoint_matches=40):
    sift_detector = cv2.SIFT_create()
    keypoints1, descriptors1 = sift_detector.detectAndCompute(image1, None)
    keypoints2, descriptors2 = sift_detector.detectAndCompute(image2, None)
    FLANN_INDEX_KDTREE = 1
    flann_index_parameters = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    flann_search_parameters = dict(checks=50)
    flann_matcher = cv2.FlannBasedMatcher(flann_index_parameters, flann_search_parameters)
    potential_matches = flann_matcher.knnMatch(descriptors1, descriptors2, k=2)
    valid_matches = []
    for first_match, second_match in potential_matches:
        if first_match.distance < 0.7 * second_match.distance:  
            valid_matches.append(first_match)
    valid_matches = sorted(valid_matches, key=lambda match: match.distance)[:max_keypoint_matches]
    matched_points = [(keypoints1[match.queryIdx].pt, keypoints2[match.trainIdx].pt) for match in valid_matches]
    source_points = np.float32([point_pair[0] for point_pair in matched_points]).reshape(-1, 1, 2)
    destination_points = np.float32([point_pair[1] for point_pair in matched_points]).reshape(-1, 1, 2)
    return np.array(matched_points), source_points, destination_points


# Computes the homography matrix from matched keypoints using the Direct Linear Transformation (DLT) method.
def compute_homography_matrix(matched_keypoints):
    num_keypoints = len(matched_keypoints)
    if num_keypoints < 4:
        raise ValueError("A minimum of 4 keypoints is required to estimate the homography matrix.")
    dlt_matrix = np.zeros((2 * num_keypoints, 9))
    for index, (point1, point2) in enumerate(matched_keypoints):
        x1, y1 = point1
        x2, y2 = point2
        dlt_matrix[2 * index] = [x1, y1, 1, 0, 0, 0, -x2 * x1, -x2 * y1, -x2]
        dlt_matrix[2 * index + 1] = [0, 0, 0, x1, y1, 1, -y2 * x1, -y2 * y1, -y2]
    _, _, V_transpose = np.linalg.svd(dlt_matrix)
    homography_matrix = V_transpose[-1].reshape(3, 3)
    return homography_matrix / homography_matrix[2, 2]


# Applies RANSAC to estimate a homography matrix from point matches robustly.
def robust_homography_estimation(point_matches, max_iterations=2000, threshold=2, subset_size=4):
    optimal_H = None
    highest_inlier_count = 0
    total_matches = len(point_matches)
    for _ in tqdm(range(max_iterations)):
        selected_indices = np.random.choice(total_matches, subset_size, replace=False)
        subset_matches = point_matches[selected_indices]
        candidate_H = compute_homography_matrix(subset_matches)
        source_points = np.array([np.append(pt1, 1) for pt1, _ in point_matches])
        target_points = np.array([np.append(pt2, 1) for _, pt2 in point_matches])
        transformed_points = candidate_H @ source_points.T
        transformed_points /= transformed_points[-1, :]
        residuals = np.linalg.norm(target_points - transformed_points.T, axis=1)
        inlier_mask = residuals < threshold
        current_inlier_count = np.sum(inlier_mask)
        if current_inlier_count > highest_inlier_count:
            highest_inlier_count = current_inlier_count
            optimal_H = candidate_H
            optimal_inliers = point_matches[inlier_mask]
    print(f"Highest number of inliers found: {highest_inlier_count}")
    return optimal_H, optimal_inliers


## Calculates the bounding box of the warped image.
def calculate_warp_bounds(H, width, height):
    corners = np.array([[0, width - 1, 0, width - 1],
                       [0, 0, height - 1, height - 1],
                       [1, 1, 1, 1]])
    transformed_corners = np.dot(H, corners)
    transformed_corners /= transformed_corners[2, :]
    x_min = int(np.min(transformed_corners[0]))
    x_max = int(np.max(transformed_corners[0]))
    y_min = int(np.min(transformed_corners[1]))
    y_max = int(np.max(transformed_corners[1]))
    return x_min, x_max, y_min, y_max


# Maps the input image onto the output image using the given homography.
def map_image(input_img, homography_matrix, output_img, use_direct_mapping=False, position_offset=(2300, 800)):
    img_height, img_width, _ = input_img.shape
    homography_inverse = np.linalg.inv(homography_matrix)
    if use_direct_mapping:
        pixel_coords = np.indices((img_width, img_height)).reshape(2, -1)
        homogeneous_coords = np.vstack((pixel_coords, np.ones(pixel_coords.shape[1])))
        mapped_coords = np.dot(homography_matrix, homogeneous_coords)
        mapped_coords /= mapped_coords[2, :]
        mapped_x, mapped_y = mapped_coords.astype(np.int32)[:2, :]
        within_bounds = (mapped_x >= 0) & (mapped_x < output_img.shape[1]) & \
                        (mapped_y >= 0) & (mapped_y < output_img.shape[0])
        mapped_x = mapped_x[within_bounds] + position_offset[0]
        mapped_y = mapped_y[within_bounds] + position_offset[1]
        input_x = pixel_coords[0][within_bounds]
        input_y = pixel_coords[1][within_bounds]
        output_img[mapped_y, mapped_x] = input_img[input_y, input_x]
    else: 
        x_min_bound, x_max_bound, y_min_bound, y_max_bound = calculate_warp_bounds(homography_matrix, img_width, img_height)
        grid_x, grid_y = np.meshgrid(np.arange(x_min_bound, x_max_bound), np.arange(y_min_bound, y_max_bound))
        coordinate_grid = np.vstack((grid_x.ravel(), grid_y.ravel(), np.ones(grid_x.size)))
        reverse_mapped_coords = np.dot(homography_inverse, coordinate_grid)
        reverse_mapped_coords /= reverse_mapped_coords[2, :]
        src_x = reverse_mapped_coords[0].astype(np.int32)
        src_y = reverse_mapped_coords[1].astype(np.int32)
        is_valid = (src_x >= 0) & (src_x < img_width) & (src_y >= 0) & (src_y < img_height)
        final_x_coords = grid_x.ravel() + position_offset[0]
        final_y_coords = grid_y.ravel() + position_offset[1]
        is_valid &= (final_x_coords >= 0) & (final_x_coords < output_img.shape[1]) & \
                    (final_y_coords >= 0) & (final_y_coords < output_img.shape[0])
        valid_indices = np.where(is_valid)[0]
        if not valid_indices.size:
            print("No valid pixels found after applying offset and boundary constraints.")
            return
        output_img[final_y_coords[valid_indices], final_x_coords[valid_indices]] = input_img[src_y[valid_indices], src_x[valid_indices]]


#   Performs image blending using Gaussian and Laplacian pyramids.
class ImagePyramidBlender:

    def __init__(self, num_levels=5):
        self.num_levels = num_levels

    # Builds a Gaussian pyramid.
    def generate_gaussian_pyramid(self, input_image):
        gauss_pyramid = [input_image]
        for _ in range(self.num_levels - 1):
            input_image = cv2.pyrDown(input_image)
            gauss_pyramid.append(input_image)
        return gauss_pyramid

    # Builds a Laplacian pyramid.
    def generate_laplacian_pyramid(self, input_image):
        laplacian_pyramid = []
        for _ in range(self.num_levels - 1):
            downscaled = cv2.pyrDown(input_image)
            upscaled = cv2.pyrUp(downscaled, dstsize=(input_image.shape[1], input_image.shape[0]))
            laplacian_layer = cv2.subtract(input_image.astype(float), upscaled.astype(float))
            laplacian_pyramid.append(laplacian_layer)
            input_image = downscaled
        laplacian_pyramid.append(input_image.astype(float)) 
        return laplacian_pyramid

    # Combines two Laplacian pyramids using a mask.
    def combine_pyramids(self, lap_pyr_a, lap_pyr_b, mask_pyr):
        combined_pyramid = []
        for idx, mask_layer in enumerate(mask_pyr):
            mask_3_channel = cv2.merge((mask_layer, mask_layer, mask_layer))
            blended_layer = lap_pyr_a[idx] * mask_3_channel + lap_pyr_b[idx] * (1 - mask_3_channel)
            combined_pyramid.append(blended_layer)
        return combined_pyramid

    # Reconstructs an image from a Laplacian pyramid.
    def reconstruct_from_pyramid(self, lap_pyr):
        reconstructed_img = lap_pyr[-1]
        for layer in reversed(lap_pyr[:-1]):
            reconstructed_img = cv2.pyrUp(reconstructed_img, dstsize=layer.shape[:2][::-1]).astype(float) + layer.astype(float)
        return reconstructed_img

    # Generates a binary mask from an image.
    def generate_mask(self, input_img):
        binary_mask = np.all(input_img != 0, axis=2)
        mask_img = np.zeros(input_img.shape[:2], dtype=float)
        mask_img[binary_mask] = 1.0
        return mask_img

    # Blends two images using pyramid blending.
    def blend_images(self, img_a, img_b):
        lap_pyr_a = self.generate_laplacian_pyramid(img_a)
        lap_pyr_b = self.generate_laplacian_pyramid(img_b)
        mask_a = self.generate_mask(img_a).astype(bool)
        mask_b = self.generate_mask(img_b).astype(bool)
        if mask_a.shape != mask_b.shape:
            min_dim = np.minimum(mask_a.shape, mask_b.shape)
            mask_a = mask_a[:min_dim[0], :min_dim[1]]
            mask_b = mask_b[:min_dim[0], :min_dim[1]]
        overlap_region = mask_a & mask_b
        coord_y, coord_x = np.where(overlap_region)
        if len(coord_x) == 0:
            split_min_x, split_max_x = img_a.shape[1] // 2, img_a.shape[1] // 2
        else:
            split_min_x, split_max_x = np.min(coord_x), np.max(coord_x)
        blend_mask = np.zeros(img_a.shape[:2])
        blend_mask[:, :(split_min_x + split_max_x) // 2] = 1.0
        mask_pyr = self.generate_gaussian_pyramid(blend_mask)
        combined_pyr = self.combine_pyramids(lap_pyr_a, lap_pyr_b, mask_pyr)
        result_img = self.reconstruct_from_pyramid(combined_pyr)
        return result_img, mask_a, mask_b


# Stitches images together to create a panorama.
class PanaromaStitcher:

    def __init__(self, max_features_count=30, match_ratio_threshold=0.75, ransac_error=2.0, 
                 pyramid_levels=6, warp_dimensions=(600, 400), image_offset_dimensions=(2300, 800)):
        self.max_features_count = max_features_count
        self.match_ratio_threshold = match_ratio_threshold
        self.ransac_error = ransac_error
        self.pyramid_levels = pyramid_levels
        self.warp_dimensions = warp_dimensions
        self.offset_dimensions = image_offset_dimensions
        self.image_file_paths = None
        self.current_scene_id = None

    # Creates a panorama from images in a directory.
    def make_panaroma_for_images_in(self, path):
        self.image_file_paths = sorted(glob.glob(os.path.join(path, '*')))
        total_images = len(self.image_file_paths)
        print(f"Found {total_images} images for stitching.")
        self.current_scene_id = self._get_scene_id(self.image_file_paths[0])
        output_directory = f'outputs/scene{self.current_scene_id}/custom'
        os.makedirs(output_directory, exist_ok=True)
        cumulative_homography = np.eye(3)  
        if total_images == 6:
            cumulative_homography = self._stitch_six()
        elif total_images == 5:
            cumulative_homography = self._stitch_five()
        elif total_images == 4:
            cumulative_homography = self._stitch_four()
        else:
            cumulative_homography = self._stitch_three()
        final_panorama_image = self._blend_all()
        cv2.imwrite(os.path.join(output_directory, 'blended_image.png'), final_panorama_image)
        return final_panorama_image, cumulative_homography
    
    # Stitches a six-image scene.
    def _stitch_six(self):
        homography_matrix = np.eye(3)
        homography_matrix = self._stitch_and_save(2, 1, homography_matrix)  
        homography_matrix = self._stitch_and_save(1, 0, homography_matrix)  
        homography_matrix = np.eye(3)
        homography_matrix = self._stitch_and_save(2, 2, homography_matrix)  
        homography_matrix = self._stitch_and_save(2, 3, homography_matrix)  
        homography_matrix = self._stitch_and_save(3, 4, homography_matrix)  
        homography_matrix = self._stitch_and_save(4, 5, homography_matrix)  
        return homography_matrix

    # Stitches a five-image scene
    def _stitch_five(self):
        homography_matrix = np.eye(3)
        homography_matrix = self._stitch_and_save(2, 1, homography_matrix)
        homography_matrix = self._stitch_and_save(1, 0, homography_matrix)
        homography_matrix = np.eye(3)
        homography_matrix = self._stitch_and_save(2, 2, homography_matrix)
        homography_matrix = self._stitch_and_save(2, 3, homography_matrix)
        homography_matrix = self._stitch_and_save(3, 4, homography_matrix)
        return homography_matrix

    # Stitches a four-image scene.
    def _stitch_four(self):
        homography_matrix = np.eye(3)
        homography_matrix = self._stitch_and_save(2, 1, homography_matrix)  
        homography_matrix = self._stitch_and_save(1, 0, homography_matrix)  
        homography_matrix = np.eye(3)
        homography_matrix = self._stitch_and_save(2, 2, homography_matrix)  
        homography_matrix = self._stitch_and_save(2, 3, homography_matrix)  
        return homography_matrix

    # Stitches a three-image scene.
    def _stitch_three(self):
        homography_matrix = np.eye(3)
        homography_matrix = self._stitch_and_save(1, 0, homography_matrix)
        homography_matrix = self._stitch_and_save(1, 1, homography_matrix)  
        return homography_matrix

    # Stitches two images and saves the result.
    def _stitch_and_save(self, source_index, destination_index, previous_homography):
        canvas_image = np.zeros((3000, 6000, 3), dtype=np.uint8)
        source_image = cv2.imread(self.image_file_paths[source_index])
        destination_image = cv2.imread(self.image_file_paths[destination_index])
        if source_image is None or destination_image is None:
            print(f"Error: Could not read image {self.image_file_paths[source_index]} or {self.image_file_paths[destination_index]}")
            return previous_homography
        print(f"Stitching: {os.path.basename(self.image_file_paths[source_index])} -> {os.path.basename(self.image_file_paths[destination_index])}")
        source_resized_image = cv2.resize(source_image, self.warp_dimensions)
        destination_resized_image = cv2.resize(destination_image, self.warp_dimensions)
        keypoint_pairs, source_points, destination_points = extract_and_match_keypoints(destination_resized_image, source_resized_image, max_keypoint_matches=self.max_features_count)
        if len(keypoint_pairs) < 4:
            print("Error: Not enough matches found.")
            return previous_homography
        homography_matrix, _ = robust_homography_estimation(keypoint_pairs, max_iterations=2000, threshold=self.ransac_error)
        if homography_matrix is None:
            return previous_homography
        if np.array_equal(previous_homography, np.eye(3)): 
            map_image(destination_resized_image, homography_matrix, canvas_image, position_offset=self.offset_dimensions)
            cumulative_homography_matrix = homography_matrix 
        else:
            cumulative_homography_matrix = np.dot(previous_homography, homography_matrix)
            map_image(destination_resized_image, cumulative_homography_matrix, canvas_image, position_offset=self.offset_dimensions)
        output_image_path = f'outputs/scene{self.current_scene_id}/custom/warped_{destination_index}.png' 
        cv2.imwrite(output_image_path, canvas_image)
        return cumulative_homography_matrix

    # Blends all warped images.
    def _blend_all(self):
        output_directory = f'outputs/scene{self.current_scene_id}/custom'
        warped_image_files = sorted(glob.glob(os.path.join(output_directory, 'warped_*.png')))
        if not warped_image_files:
            raise ValueError("No warped images found for blending.")
        image_blender = ImagePyramidBlender(num_levels=self.pyramid_levels)
        composite_image = cv2.imread(warped_image_files[0])
        for warped_file in warped_image_files[1:]:
            print(f"Blending: {os.path.basename(warped_file)}")
            image_to_blend = cv2.imread(warped_file)
            if image_to_blend is not None:
                composite_image, _, _ = image_blender.blend_images(composite_image, image_to_blend) 
        return composite_image

    # Extracts the scene ID from the file path.
    def _get_scene_id(self, file_path):
        directory_name = os.path.basename(os.path.dirname(file_path))
        scene_identifier = ''.join(filter(str.isdigit, directory_name))
        return int(scene_identifier) if scene_identifier.isdigit() else 1