import cv2
import numpy as np

def draw_matches(img1, keypoints1, img2, keypoints2, matcher, inliers=None):
    rows1, cols1, _ = img1.shape
    rows2, cols2, _ = img2.shape
    out = np.zeros((max(rows1, rows2), cols1 + cols2, 3), dtype='uint8')
    out[:rows1, :cols1, :] = img1
    out[:rows2, cols1:cols1 + cols2, :] = img2

    for match in matcher:
        mat = match[0]
        img1_idx, img2_idx = mat.queryIdx, mat.trainIdx
        (x1, y1), (x2, y2) = keypoints1[img1_idx].pt, keypoints2[img2_idx].pt

        inlier = False
        if inliers is not None:
            for i in inliers:
                if np.array_equal(i[:2], [x1, y1]) and np.array_equal(i[2:], [x2, y2]):
                    inlier = True

        cv2.circle(out, (int(x1), int(y1)), 4, (255, 0, 0), 1)
        cv2.circle(out, (int(x2) + cols1, int(y2)), 4, (255, 0, 0), 1)

        if inliers is not None and inlier:
            cv2.line(out, (int(x1), int(y1)), (int(x2) + cols1, int(y2)), (0, 255, 0), 1)
        elif inliers is not None:
            cv2.line(out, (int(x1), int(y1)), (int(x2) + cols1, int(y2)), (0, 0, 255), 1)

        if inliers is None:
            cv2.line(out, (int(x1), int(y1)), (int(x2) + cols1, int(y2)), (255, 0, 0), 1)

    return out

def find_matches(descriptors1, descriptors2, ratio=0.75):
    matches = []
    for i in range(len(descriptors1)):
        distances = np.linalg.norm(descriptors2 - descriptors1[i], axis=1)
        min_index = np.argmin(distances)
        first_min_distance = distances[min_index]

        distances[min_index] = np.inf
        second_min_distance = np.min(distances)

        if first_min_distance < ratio * second_min_distance:
            matches.append(cv2.DMatch(_queryIdx=i, _trainIdx=min_index, _distance=first_min_distance))
    return matches

def generate_random(src_pts, dest_pts, N):
    r = np.random.choice(len(src_pts), N)
    src = [src_pts[i] for i in r]
    dest = [dest_pts[i] for i in r]
    return np.asarray(src, dtype=np.float32), np.asarray(dest, dtype=np.float32)

def find_homography(src, dest, N):
    A = []
    for i in range(N):
        x, y = src[i][0][0], src[i][0][1]
        xp, yp = dest[i][0][0], dest[i][0][1]
        A.extend([
            [x, y, 1, 0, 0, 0, -x * xp, -xp * y, -xp],
            [0, 0, 0, x, y, 1, -yp * x, -yp * y, -yp]
        ])
    A = np.asarray(A)
    _, _, Vh = np.linalg.svd(A)
    L = Vh[-1, :] / Vh[-1, -1]
    H = L.reshape(3, 3)
    return H

def ransac_homography(src_pts, dest_pts):
    max_inliers = 0
    max_src_lines = []
    max_dest_lines = []

    for _ in range(70):
        src_p, dest_p = generate_random(src_pts, dest_pts, 4)
        H = find_homography(src_p, dest_p, 4)
        inliers = 0
        lines_src = []
        lines_dest = []

        for p1, p2 in zip(src_pts, dest_pts):
            p1_u = np.append(p1, 1).reshape(3, 1)
            p2_e = H.dot(p1_u)
            p2_e = (p2_e / p2_e[2])[:2].reshape(1, 2)[0]

            if cv2.norm(p2 - p2_e) < 10:
                inliers += 1
                lines_src.append(p1)
                lines_dest.append(p2)

        if inliers > max_inliers:
            max_inliers = inliers
            max_src_lines = lines_src.copy()
            max_src_lines = np.asarray(max_src_lines, dtype=np.float32)
            max_dest_lines = lines_dest.copy()
            max_dest_lines = np.asarray(max_dest_lines, dtype=np.float32)

    H_final = find_homography(max_src_lines, max_dest_lines, max_inliers)
    return H_final

def find_canvas_size(image1, img2, homography):
    h1, w1 = image1.shape[:2]
    corners1 = np.array([[0, 0], [0, h1], [w1, h1], [w1, 0]], dtype=np.float32).reshape(-1, 1, 2)
    h2, w2 = img2.shape[:2]
    corners2 = np.array([[0, 0], [0, h2], [w2, h2], [w2, 0]], dtype=np.float32).reshape(-1, 1, 2)
    homography = homography.astype(np.float32)
    transformed_corners2 = cv2.perspectiveTransform(corners2, homography)
    all_corners = np.concatenate((corners1, transformed_corners2), axis=0)
    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel())
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel())
    canvas_width = x_max - x_min
    canvas_height = y_max - y_min
    translation = [-x_min, -y_min]
    return (canvas_width, canvas_height), translation

def warp_perspective_custom(img, homography, output_size):
    height, width = img.shape[:2]
    output_img = np.zeros((output_size[1], output_size[0], img.shape[2]), dtype=img.dtype)

    for y_out in range(output_size[1]):
        for x_out in range(output_size[0]):
            homog_coord = np.dot(np.linalg.inv(homography), np.array([x_out, y_out, 1]))
            x_in, y_in, w_in = homog_coord / homog_coord[2]

            x_in_int, y_in_int = int(x_in), int(y_in)

            if 0 <= x_in_int < width - 1 and 0 <= y_in_int < height - 1:
                dx = x_in - x_in_int
                dy = y_in - y_in_int

                pixel1 = (1 - dx) * img[y_in_int, x_in_int] + dx * img[y_in_int, x_in_int + 1]
                pixel2 = (1 - dx) * img[y_in_int + 1, x_in_int] + dx * img[y_in_int + 1, x_in_int + 1]

                output_img[y_out, x_out] = (1 - dy) * pixel1 + dy * pixel2

    return output_img

def apply_homography(homography, x, y):
    homogenous = np.array([x, y, 1])
    transformed = homography @ homogenous
    cartesian = transformed[:2] / transformed[2]
    return cartesian

def warp_perspective_custom(image, homography, output):
    height, width = image.shape[:2]
    corners = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1],
    ])
    transformed_corners = np.array([apply_homography(homography, pt[0], pt[1]) for pt in corners])
    x_min, y_min = np.min(transformed_corners, axis=0)
    x_max, y_max = np.max(transformed_corners, axis=0)
    translation = np.array([
        [1, 0, -x_min],
        [0, 1, -y_min],
        [0, 0, 1]
    ])
    homography = translation @ homography
    output = (int(y_max - y_min), int(x_max - x_min))
    warped_image = np.zeros((output[0], output[1], image.shape[2]), dtype=image.dtype)
    homography_inv = np.linalg.inv(homography)

    for y in range(output[0]):
        for x in range(output[1]):
            source_x, source_y = apply_homography(homography_inv, x, y)
            if 0 <= source_x < width and 0 <= source_y < height:
                source_x, source_y = int(source_x), int(source_y)
                warped_image[y, x] = image[source_y, source_x]

    return warped_image

def stitch_images(img1, img2):
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)
    good_matches = find_matches(descriptors1, descriptors2)
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    homography = ransac_homography(src_pts, dst_pts)
    (canvas_width, canvas_height), translation = find_canvas_size(img1, img2, homography)
    homography_translated = np.array([
        [1, 0, translation[0]],
        [0, 1, translation[1]],
        [0, 0, 1]
    ]) @ homography
    warped_image1 = warp_perspective_custom(img1, homography_translated, (canvas_height, canvas_width))
    output_image = np.zeros((canvas_height, canvas_width, 3), dtype=warped_image1.dtype)
    x_offset, y_offset = translation
    overlap_width = max(0, min(warped_image1.shape[1] - x_offset, img2.shape[1]))
    alpha_mask = np.zeros((img2.shape[0], overlap_width), dtype=np.float32)
    alpha_mask[:, :] = np.linspace(1, 0, overlap_width)

    for c in range(3):
        output_image[y_offset:y_offset + img2.shape[0], x_offset:x_offset + overlap_width, c] = (
                alpha_mask * warped_image1[y_offset:y_offset + img2.shape[0], x_offset:x_offset + overlap_width, c] +
                (1 - alpha_mask) * img2[:, :overlap_width, c]
        )
    output_image[y_offset:y_offset + img2.shape[0], x_offset + overlap_width:x_offset + img2.shape[1]] = img2[:, overlap_width:]

    output_image[:warped_image1.shape[0], :warped_image1.shape[1]] = np.where(
        output_image[:warped_image1.shape[0], :warped_image1.shape[1]] == 0,
        warped_image1,
        output_image[:warped_image1.shape[0], :warped_image1.shape[1]]
    )

    return output_image

if __name__ == "__main__":
    image1 = cv2.imread("im1_1.png")
    image2 = cv2.imread("im1_2.png")
    result_image = stitch_images(image1, image2)
    cv2.imwrite('stitched_image.jpg', result_image)
    print('Stitched image saved as stitched_image.jpg')
