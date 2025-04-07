import cv2
import numpy as np
from scipy.ndimage import interpolation as inter

def correct_skew(image, delta=0.5, limit=15, debug=False):
    """Correct skew of the image using projection profile method"""
    
    def determine_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1, dtype=float)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)
        return score

    if image is None:
        raise ValueError("Input image is empty or not loaded properly.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    scores = []
    angles = np.arange(-limit, limit + delta, delta)

    for angle in angles:
        score = determine_score(thresh, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]

    if debug:
        import matplotlib.pyplot as plt
        plt.plot(angles, scores)
        plt.title("Skew Angle vs Score")
        plt.xlabel("Angle (degrees)")
        plt.ylabel("Score")
        plt.grid(True)
        plt.show()

    print(f"[INFO] Best skew angle: {best_angle:.2f}")

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    corrected = cv2.warpAffine(image, M, (w, h),
                               flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)

    return best_angle, corrected
