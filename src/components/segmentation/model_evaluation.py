import os
import torch
import numpy as np
from scipy import ndimage

from src.logger import logging
from src.exception import CustomException
from src.components.segmentation.model import Model
from src.entity.config_entity import ModelEvaluationConfig

SIZE_BUCKETS = [(10, 50), (50, 200), (200, float('inf'))]


def pixel_confusion_counts(pred_binary: np.ndarray, target: np.ndarray) -> tuple:
    """Accumulate TP/FP/FN/TN pixel counts for one batch. Sum these across the
    whole test set before computing precision/recall/dice — computing dice per
    batch and averaging batches is biased when batches have very different
    tumor pixel counts (a batch of all no_tumor slices would silently drag the
    average around)."""
    pred_binary = pred_binary.astype(bool)
    target = target.astype(bool)

    tp = np.logical_and(pred_binary, target).sum()
    fp = np.logical_and(pred_binary, ~target).sum()
    fn = np.logical_and(~pred_binary, target).sum()
    tn = np.logical_and(~pred_binary, ~target).sum()
    return tp, fp, fn, tn


def component_recall_by_size(pred_binary: np.ndarray, target: np.ndarray, buckets=SIZE_BUCKETS) -> dict:
    """For each ground-truth lesion (connected component), was it detected —
    i.e. does the prediction overlap it by at least one pixel — bucketed by
    lesion pixel area. Answers 'is this a small-lesion-specific problem?'
    rather than hiding it in one aggregate number."""
    results = {b: [0, 0] for b in buckets}  # [detected, total]

    for i in range(target.shape[0]):
        labeled_gt, n = ndimage.label(target[i, 0])
        for comp_id in range(1, n + 1):
            mask = (labeled_gt == comp_id)
            size = mask.sum()
            if size < buckets[0][0]:
                continue
            for lo, hi in buckets:
                if lo <= size < hi:
                    results[(lo, hi)][1] += 1
                    if (pred_binary[i, 0] * mask).sum() > 0:
                        results[(lo, hi)][0] += 1
                    break
    return results


def component_precision(pred_binary: np.ndarray, target: np.ndarray, min_size: int = 10) -> tuple:
    """The flip side of recall: of everything the model predicted as a lesion,
    how much of it corresponds to a real ground-truth lesion vs. is a spurious
    blob with no basis in the mask? A recall-biased loss can inflate recall by
    over-predicting — this is what catches that trade-off."""
    real_blobs, total_blobs = 0, 0

    for i in range(target.shape[0]):
        labeled_pred, n = ndimage.label(pred_binary[i, 0])
        for comp_id in range(1, n + 1):
            mask = (labeled_pred == comp_id)
            if mask.sum() < min_size:
                continue
            total_blobs += 1
            if (target[i, 0] * mask).sum() > 0:
                real_blobs += 1
    return real_blobs, total_blobs


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig, test_loader):
        self.config = config
        self.test_loader = test_loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'Testing using device: {self.device}')

        # This threshold must already be tuned on the VAL set, not test — sweeping
        # thresholds against test and reporting the best one is a form of leakage
        # (you'd be reporting a number partially fit to the test set itself).
        self.threshold = self.config.threshold

    def evaluate(self):
        try:
            logging.info('Loading model')
            model = Model(freeze_layers=False).to(self.device)
            model.load_state_dict(torch.load(self.config.model_path, map_location=self.device))
            model.eval()
            logging.info('Model loaded')

            tp_total = fp_total = fn_total = tn_total = 0
            size_bucket_totals = {b: [0, 0] for b in SIZE_BUCKETS}
            blob_real_total = blob_count_total = 0

            with torch.no_grad():
                for image, mask in self.test_loader:
                    image, mask = image.to(self.device), mask.to(self.device)

                    output = model(image)
                    probs = torch.sigmoid(output)
                    pred_binary = (probs > self.threshold).float().cpu().numpy()
                    mask_np = mask.cpu().numpy()

                    tp, fp, fn, tn = pixel_confusion_counts(pred_binary, mask_np)
                    tp_total += tp
                    fp_total += fp
                    fn_total += fn
                    tn_total += tn

                    batch_buckets = component_recall_by_size(pred_binary, mask_np)
                    for b, (detected, total) in batch_buckets.items():
                        size_bucket_totals[b][0] += detected
                        size_bucket_totals[b][1] += total

                    real, count = component_precision(pred_binary, mask_np)
                    blob_real_total += real
                    blob_count_total += count

            eps = 1e-6
            dice = (2 * tp_total + eps) / (2 * tp_total + fp_total + fn_total + eps)
            iou = (tp_total + eps) / (tp_total + fp_total + fn_total + eps)
            pixel_precision = (tp_total + eps) / (tp_total + fp_total + eps)
            pixel_recall = (tp_total + eps) / (tp_total + fn_total + eps)

            overall_detected = sum(v[0] for v in size_bucket_totals.values())
            overall_lesions = sum(v[1] for v in size_bucket_totals.values())
            component_recall_overall = overall_detected / overall_lesions if overall_lesions > 0 else 0.0
            component_precision_overall = blob_real_total / blob_count_total if blob_count_total > 0 else 0.0

            logging.info(f'Threshold used: {self.threshold}')
            logging.info(f'Pixel Dice: {dice:.4f} | Pixel IoU: {iou:.4f} | '
                         f'Pixel Precision: {pixel_precision:.4f} | Pixel Recall: {pixel_recall:.4f}')
            logging.info(f'Component Recall (overall): {component_recall_overall:.4f} '
                         f'({overall_detected}/{overall_lesions})')
            logging.info(f'Component Precision (overall): {component_precision_overall:.4f} '
                         f'({blob_real_total}/{blob_count_total} predicted blobs correspond to a real lesion)')

            for (lo, hi), (detected, total) in size_bucket_totals.items():
                label = f'{lo}-{hi}px' if hi != float('inf') else f'{lo}px+'
                recall = detected / total if total > 0 else float('nan')
                logging.info(f'Component Recall [{label}]: {recall:.4f} ({detected}/{total})')

            return {
                'dice': dice,
                'iou': iou,
                'pixel_precision': pixel_precision,
                'pixel_recall': pixel_recall,
                'component_recall_overall': component_recall_overall,
                'component_precision_overall': component_precision_overall,
                'component_recall_by_size': size_bucket_totals,
            }

        except Exception as e:
            logging.exception('Error occurred at ModelEvaluation.evaluate')
            raise CustomException(e)