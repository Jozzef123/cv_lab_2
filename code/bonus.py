"""Estimate and correct radial camera distortion for the Bonus report section."""

from __future__ import annotations

import sys

import cv2
import numpy as np

import gt
import utils
import viz
from features import features, features_sift, match
from fitting import fit_homography, ransac
from odometry import chain


def undistort_points(points: np.ndarray, kappa: float, frame_size: tuple[int, int]) -> np.ndarray:
    """Invert the radial model for distorted (x, y) keypoints with Newton steps."""
    width, height = frame_size
    center = np.array([width / 2.0, height / 2.0], dtype=np.float64)
    offsets = np.asarray(points, dtype=np.float64).reshape(-1, 2) - center
    radius_d = np.linalg.norm(offsets, axis=1)
    radius = radius_d.copy()
    scale2 = 600.0**2
    for _ in range(8):
        value = radius * (1.0 + kappa * radius**2 / scale2) - radius_d
        derivative = 1.0 + 3.0 * kappa * radius**2 / scale2
        radius -= value / np.where(np.abs(derivative) > 1e-8, derivative, 1e-8)
        radius = np.maximum(radius, 0.0)
    factor = np.divide(radius, radius_d, out=np.ones_like(radius), where=radius_d > 1e-8)
    return center + offsets * factor[:, None]


def undistort_image(image: np.ndarray, kappa: float) -> np.ndarray:
    """Remap an ideal output pixel to its corresponding distorted input pixel."""
    height, width = image.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    center_x, center_y = width / 2.0, height / 2.0
    radius2 = (xx - center_x) ** 2 + (yy - center_y) ** 2
    radial = 1.0 + float(kappa) * radius2 / (600.0**2)
    map_x = center_x + (xx - center_x) * radial
    map_y = center_y + (yy - center_y) * radial
    return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _feature_pair_matches(
    feature_a: tuple[np.ndarray, np.ndarray],
    feature_b: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Return matched keypoint coordinates for two cached feature sets."""
    points_a, descriptors_a = feature_a
    points_b, descriptors_b = feature_b
    pairs = match(descriptors_a, descriptors_b)
    return points_a[pairs[:, 0]], points_b[pairs[:, 1]]


def _calibration_pairs(
    frames: list[np.ndarray],
    frame_features: list[tuple[np.ndarray, np.ndarray]],
    pair_count: int = 10,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Select consecutive pairs spread over a flight, excluding weak matches."""
    starts = np.linspace(0, len(frames) - 2, min(pair_count, len(frames) - 1), dtype=int)
    selected: list[tuple[np.ndarray, np.ndarray]] = []
    for index in np.unique(starts):
        points_a, points_b = _feature_pair_matches(frame_features[index], frame_features[index + 1])
        if len(points_a) >= 12:
            selected.append((points_a, points_b))
    return selected


def _distortion_cost(
    pairs: list[tuple[np.ndarray, np.ndarray]],
    kappa: float,
    frame_size: tuple[int, int],
    seed: int = 19,
) -> float:
    """Score kappa by truncated squared homography residuals, without ground truth."""
    total_loss, total_points = 0.0, 0
    for pair_index, (points_a, points_b) in enumerate(pairs):
        ideal_a = undistort_points(points_a, kappa, frame_size)
        ideal_b = undistort_points(points_b, kappa, frame_size)
        transform, inliers, _ = ransac(
            ideal_a,
            ideal_b,
            fit_homography,
            sample=4,
            thresh=1.5,
            p_success=0.99,
            max_iter=120,
            seed=seed + pair_index,
        )
        if transform is None or inliers is None:
            losses = np.full(len(ideal_a), 25.0, dtype=np.float64)
        else:
            projected = utils.apply(transform, ideal_a)
            residual2 = np.sum((projected - ideal_b) ** 2, axis=1)
            losses = np.minimum(residual2, 25.0)
        total_loss += float(losses.sum())
        total_points += len(losses)
    return total_loss / max(total_points, 1)


def estimate_kappa(
    pairs: list[tuple[np.ndarray, np.ndarray]],
    frame_size: tuple[int, int],
) -> tuple[float, np.ndarray, np.ndarray]:
    """Search the requested range coarsely, then refine around its best value."""
    coarse = np.linspace(-0.6, 0.6, 17)
    coarse_costs = np.array([_distortion_cost(pairs, value, frame_size) for value in coarse])
    best_coarse = float(coarse[int(np.argmin(coarse_costs))])
    step = float(coarse[1] - coarse[0])
    fine = np.linspace(max(-0.6, best_coarse - step), min(0.6, best_coarse + step), 17)
    fine_costs = np.array([_distortion_cost(pairs, value, frame_size) for value in fine])
    best = int(np.argmin(fine_costs))
    grid = np.r_[coarse, fine]
    costs = np.r_[coarse_costs, fine_costs]
    return float(fine[best]), grid, costs


def _chain_scores(
    flight_path: str,
    frame_features: list[tuple[np.ndarray, np.ndarray]],
) -> dict[str, tuple[list[np.ndarray], list[np.ndarray | None], list[int], list[int], float, float]]:
    """Chain similarity and homography models and score them against the truth."""
    scores = {}
    for model in ("similarity", "homography"):
        transforms, steps, lost, inliers = chain(frame_features, model)
        errors = gt.path_errors(flight_path, transforms)
        scores[model] = (transforms, steps, lost, inliers, float(errors.mean()), float(errors[-1]))
    return scores


def _save_example(frame: np.ndarray, kappa: float, output_path: str) -> None:
    """Save an original/undistorted pair for a visual straight-line comparison."""
    corrected = undistort_image(frame, kappa)
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        corrected = cv2.cvtColor(corrected, cv2.COLOR_GRAY2BGR)
    comparison = np.hstack((frame, corrected))
    cv2.putText(comparison, "original", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(comparison, "undistorted", (frame.shape[1] + 16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    if not cv2.imwrite(output_path, comparison):
        raise OSError(f"Could not save undistortion example: {output_path}")


def main() -> None:
    """Estimate distortion, compare paths, and write all Bonus report artifacts."""
    use_sift = "--sift" in sys.argv[1:]
    feature_fn = features_sift if use_sift else features
    output = utils.results_dir("bonus")
    frame_size = (640, 480)
    flight_names = ("corniche", "loop", "lost_at_sea")
    raw_frames: dict[str, list[np.ndarray]] = {}
    raw_features: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}

    # Original-frame matches are the only measurements used to estimate kappa.
    for name in flight_names:
        print(f"... loading original frames and features: {name}", flush=True)
        raw_frames[name] = utils.load_frames(utils.flight(name))
        raw_features[name] = [feature_fn(image) for image in raw_frames[name]]

    kappa_values: dict[str, float] = {}
    cost_curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    pair_counts: dict[str, int] = {}
    for name in flight_names:
        print(f"... estimating kappa from {name} matches", flush=True)
        pairs = _calibration_pairs(raw_frames[name], raw_features[name])
        if not pairs:
            raise RuntimeError(f"No usable consecutive-frame matches for kappa estimation on {name}")
        pair_counts[name] = len(pairs)
        kappa, grid, costs = estimate_kappa(pairs, frame_size)
        kappa_values[name] = kappa
        cost_curves[name] = (grid, costs)
        print(f"    {name}: kappa={kappa:.4f}, cost={costs[np.argmin(costs)]:.4f}px^2 over {len(pairs)} pairs")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 4.5))
    for name, (grid, costs) in cost_curves.items():
        order = np.argsort(grid)
        axis.plot(grid[order], costs[order], "-o", markersize=3, label=f"{name} (best {kappa_values[name]:.3f})")
    axis.axvline(0.0, color="black", linewidth=0.8, linestyle="--")
    axis.set_xlabel("kappa")
    axis.set_ylabel("mean truncated squared homography residual (px^2)")
    axis.set_title("Radial distortion estimate from frame matches (no GT)")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(f"{output}/kappa_cost.png", dpi=150)
    plt.close(figure)

    kappa_rows = []
    for name in flight_names:
        grid, costs = cost_curves[name]
        best_idx = int(np.argmin(costs))
        kappa_rows.append([
            name,
            kappa_values[name],
            float(costs[best_idx]),
            f"{pair_counts[name]} spread consecutive pairs",
        ])
    utils.table(
        f"{output}/kappa.md",
        ["flight", "estimated kappa", "cost (px^2)", "matches used"],
        kappa_rows,
    )

    example_frame = utils.load_frames(utils.flight("loop"), color=True)[20]
    _save_example(example_frame, kappa_values["loop"], f"{output}/undistort_example.jpg")

    raw_scores: dict[str, dict[str, tuple[list[np.ndarray], list[np.ndarray | None], list[int], list[int], float, float]]] = {}
    corrected_features = dict(raw_features)
    corrected_kappas = {"corniche": 0.0, "loop": kappa_values["loop"], "lost_at_sea": kappa_values["lost_at_sea"]}
    for name in flight_names:
        print(f"... chaining raw {name}", flush=True)
        raw_scores[name] = _chain_scores(utils.flight(name), raw_features[name])
        if name != "corniche":
            print(f"... undistorting and re-extracting features: {name}", flush=True)
            corrected_images = [undistort_image(image, corrected_kappas[name]) for image in raw_frames[name]]
            corrected_features[name] = [feature_fn(image) for image in corrected_images]

    rows = []
    corrected_scores = {}
    for name in flight_names:
        corrected_scores[name] = _chain_scores(utils.flight(name), corrected_features[name])
        for label, scores in (("raw", raw_scores[name]), ("undistorted", corrected_scores[name])):
            for model in ("similarity", "homography"):
                transforms, steps, lost, inliers, mean_error, final_error = scores[model]
                rows.append([name, label, model, len(transforms), len(lost), mean_error, final_error])
                np.save(f"{output}/T_{name}_{label}_{model}.npy", np.asarray(transforms))
        runs = {}
        for model in ("similarity", "homography"):
            runs[f"raw {model}"] = raw_scores[name][model][0]
            runs[f"undistorted {model}"] = corrected_scores[name][model][0]
        viz.plot_paths(utils.flight(name), runs, f"{output}/paths_{name}.png")

    utils.table(
        f"{output}/q3_2_models_bonus.md",
        ["flight", "frame state", "model", "frames", "lost steps", "mean error (m)", "final error (m)"],
        rows,
    )

    # Compare GT reprojection errors only after the distortion parameters are fixed.
    before_pairs, after_pairs = [], []
    radius_pairs = (("corniche", 20, 21), ("loop", 20, 21), ("loop", 3, 132),
                    ("lost_at_sea", 20, 21), ("lost_at_sea", 40, 41))
    for name, index_a, index_b in radius_pairs:
        true_transform = gt.gt_H(utils.flight(name), index_a, index_b)
        points_a, points_b = _feature_pair_matches(raw_features[name][index_a], raw_features[name][index_b])
        before_pairs.append((f"{name} {index_a}-{index_b} before", points_a, points_b, true_transform))
        if name == "corniche":
            corrected_a, corrected_b = points_a, points_b
        else:
            corrected_a, corrected_b = _feature_pair_matches(
                corrected_features[name][index_a], corrected_features[name][index_b]
            )
        after_pairs.append((f"{name} {index_a}-{index_b} after", corrected_a, corrected_b, true_transform))
    viz.plot_error_vs_radius(before_pairs + after_pairs, f"{output}/error_vs_radius_before_after.png")
    print(f"\nBonus outputs saved to {output}")
    if use_sift:
        print("Feature front end: OpenCV SIFT (--sift).")


if __name__ == "__main__":
    utils.run(main)