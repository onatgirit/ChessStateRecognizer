import numpy as np
import cv2
from ChessboardConfiguration import ChessboardConfiguration as cfg


class ContourExtractor:
    @staticmethod
    def get_approx_quad(model_output):
        ret, thresh = cv2.threshold(model_output, 127, 255, cv2.THRESH_BINARY_INV)
        contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        biggest_contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]
        accuracy = 0.03 * cv2.arcLength(biggest_contour, True)
        approx = cv2.approxPolyDP(biggest_contour, accuracy, True)
        return approx

    @staticmethod
    def get_warped_board(img, approx):
        s = np.float32(approx.sum(axis=1))
        dst = np.array([
            [0, 0],
            [cfg.EXTRACTED_BOARD_RESOLUTION[1] - 1, 0],
            [cfg.EXTRACTED_BOARD_RESOLUTION[1] - 1, cfg.EXTRACTED_BOARD_RESOLUTION[0] - 1],
            [0, cfg.EXTRACTED_BOARD_RESOLUTION[0] - 1]], dtype="float32")
        transformation_matrix = cv2.getPerspectiveTransform(s, dst)
        warped = cv2.warpPerspective(img, transformation_matrix, cfg.EXTRACTED_BOARD_RESOLUTION)
        return warped
