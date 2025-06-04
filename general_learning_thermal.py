!pip install transformers open3d scikit-learn opencv-python pandas tqdm psutil hdbscan -q
from google.colab.patches import cv2_imshow
import cv2
import numpy as np
import torch
import open3d as o3d
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from transformers import CLIPProcessor, CLIPModel, logging
from PIL import Image, ImageEnhance
import os
import hdbscan
from tqdm.auto import tqdm
import pandas as pd
import gc
import psutil
import torch.nn.functional as F

# Suppress warnings and optimize performance
logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = "cuda" if torch.cuda.is_available() else "cpu"
use_amp = torch.cuda.is_available()
torch.backends.cudnn.benchmark = True
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Memory monitoring function
def check_memory():
    process = psutil.Process()
    mem_info = process.memory_info()
    ram_usage = mem_info.rss / 1024**2
    if device == "cuda":
        gpu_mem = torch.cuda.memory_allocated() / 1024**2
        print(f"RAM: {ram_usage:.2f} MB, GPU: {gpu_mem:.2f} MB")
    else:
        print(f"RAM: {ram_usage:.2f} MB")
    return ram_usage

# --------------------------------------------
# 🎞️ Step 1: Load Drone Thermal Video
# --------------------------------------------
def load_drone_video(video_path, width=240, height=180):
    print(f"Loading drone thermal video from {video_path}...")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    for i in tqdm(range(total_frames), desc="Extracting frames"):
        success, frame = cap.read()
        if not success:
            break
        if i % 5 == 0:  # Sample keyframes
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray)
    cap.release()
    print(f"✅ Loaded {len(frames)} key thermal frames.")
    check_memory()
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return frames, fps

# --------------------------------------------
# 🖼️ Step 2: Generate Initial Labels (Automated)
# --------------------------------------------
def generate_initial_labels(frames, normalized_scores, output_csv):
    print("Generating initial labels automatically...")
    check_memory()

    # Heuristic: Intensity-based labeling with adaptive threshold
    intensity_labels = []
    for i, frame in enumerate(frames):
        intensity_threshold = np.mean(frame) + 2 * np.std(frame)
        label = 1 if np.mean(frame > intensity_threshold) > 0.25 else 0
        intensity_labels.append((i, label))

    # Weighted temporal smoothing
    smoothed_labels = []
    window_size = 7
    weights = np.array([0.1, 0.2, 0.4, 0.6, 0.4, 0.2, 0.1])[:window_size]
    weights /= weights.sum()
    for i in range(len(intensity_labels)):
        start = max(0, i - window_size // 2)
        end = min(len(intensity_labels), i + window_size // 2 + 1)
        window_labels = [intensity_labels[j][1] for j in range(start, end)]
        padded_weights = np.zeros(len(window_labels))
        padded_weights[:min(len(weights), len(window_labels))] = weights[:len(window_labels)]
        padded_weights /= padded_weights.sum()
        smoothed_label = 1 if np.sum(np.array(window_labels) * padded_weights) > 0.5 else 0
        smoothed_labels.append((intensity_labels[i][0], smoothed_label))
    automated_labels = smoothed_labels

    # Cluster frames to refine labels
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model.eval()
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(frames), 2):
            batch_frames = frames[i:i+2]
            batch_images = [Image.fromarray(cv2.equalizeHist(frame)) for frame in batch_frames]
            inputs = processor(images=batch_images, return_tensors="pt", padding=True).to(device)
            image_features = model.get_image_features(**inputs).cpu().numpy()
            embeddings.extend(image_features)
    embeddings = np.array(embeddings)
    del model, processor
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    clustering = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=3, cluster_selection_epsilon=0.7)
    clustering.fit(embeddings)
    cluster_labels = clustering.labels_

    # Assign labels based on cluster scores
    cluster_scores = {}
    for cluster in np.unique(cluster_labels[cluster_labels != -1]):
        cluster_indices = np.where(cluster_labels == cluster)[0]
        cluster_mean_score = np.mean([normalized_scores[i] for i in cluster_indices])
        cluster_scores[cluster] = 1 if cluster_mean_score > 0.5 else 0

    final_labels = []
    for i in range(len(frames)):
        cluster = cluster_labels[i]
        if cluster != -1 and cluster in cluster_scores:
            final_labels.append((i, cluster_scores[cluster]))
        else:
            for idx, label in automated_labels:
                if idx == i:
                    final_labels.append((idx, label))
                    break

    # Downsample normal frames to balance classes
    normal_indices = [idx for idx, label in final_labels if label == 0]
    anomaly_indices = [idx for idx, label in final_labels if label == 1]
    if len(normal_indices) > len(anomaly_indices):
        normal_indices = np.random.choice(normal_indices, size=len(anomaly_indices), replace=False)
    final_labels = [(idx, 1) for idx in anomaly_indices] + [(idx, 0) for idx in normal_indices]

    # Combine and save labels
    unique_labels = {idx: label for idx, label in final_labels}
    df = pd.DataFrame([(idx, label) for idx, label in unique_labels.items()], columns=['frame_index', 'label'])
    df.to_csv(output_csv, index=False)
    print(f"✅ Saved {len(df)} labels to {output_csv}")

    # Check label distribution
    label_counts = df['label'].value_counts()
    print(f"Label distribution: {label_counts.to_dict()}")
    check_memory()
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return df

# --------------------------------------------
# 🤖 Step 3: CLIP Model Processing
# --------------------------------------------
def process_with_clip(frames, batch_size=2):
    print("Processing with CLIP model...")
    if device == "cuda":
        torch.cuda.empty_cache()
        gc.collect()
    check_memory()

    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model.eval()

    prompts = ["intense thermal anomaly in concrete curing", "normal concrete curing temperature profile"]
    text_inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)

    clip_scores = []
    with torch.no_grad():
        for i in tqdm(range(0, len(frames), batch_size)):
            batch_frames = frames[i:i+batch_size]
            batch_pil_images = [Image.fromarray(cv2.equalizeHist(frame)) for frame in batch_frames]
            with torch.amp.autocast('cuda', enabled=use_amp):
                image_inputs = processor(images=batch_pil_images, return_tensors="pt", padding=True).to(device)
                outputs = model(**text_inputs, **image_inputs)
                logits_per_image = outputs.logits_per_image.softmax(dim=1)
                batch_scores = logits_per_image[:, 0].cpu().numpy().tolist()
                clip_scores.extend(batch_scores)

    min_score, max_score = min(clip_scores), max(clip_scores)
    normalized_scores = [(s - min_score) / (max_score - min_score) if max_score > min_score else s for s in clip_scores]

    intensity_scores = [np.mean(frame > (np.mean(frame) + 2 * np.std(frame))) for frame in frames]
    intensity_scores = [(s - min(intensity_scores)) / (max(intensity_scores) - min(intensity_scores)) for s in intensity_scores]
    normalized_scores = [(0.7 * clip + 0.3 * intensity) for clip, intensity in zip(normalized_scores, intensity_scores)]

    del model, processor
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    print(f"✅ Processed {len(frames)} frames.")
    check_memory()
    return clip_scores, normalized_scores

# --------------------------------------------
# 📊 Step 4: Classification and Metrics
# --------------------------------------------
def optimize_threshold(normalized_scores, ground_truth_labels):
    if ground_truth_labels is None or len(np.unique(ground_truth_labels)) < 2:
        print("Insufficient label variety. Using default threshold 0.5.")
        return 0.5, [int(score > 0.5) for score in normalized_scores]

    best_score, best_threshold = 0, 0.5
    for threshold in np.linspace(0.6, 0.95, 100):
        predicted_labels = [int(score > threshold) for score in normalized_scores]
        f1 = f1_score(ground_truth_labels, predicted_labels, zero_division=0)
        precision = precision_score(ground_truth_labels, predicted_labels, zero_division=0)
        recall = recall_score(ground_truth_labels, predicted_labels, zero_division=0)
        score = 0.5 * precision + 0.5 * recall
        if score > best_score:
            best_score, best_threshold = score, threshold

    print(f"Optimized threshold: {best_threshold:.4f} (Score: {best_score:.4f})")
    return best_threshold, [int(score > best_threshold) for score in normalized_scores]

def compute_metrics(ground_truth_labels, predicted_labels, normalized_scores):
    if ground_truth_labels is None or len(np.unique(ground_truth_labels)) < 2:
        print("Insufficient label variety. Skipping metric computation.")
        return None, None, None, None, None

    n_positive = sum(1 for gt in ground_truth_labels if gt == 1)
    n_negative = sum(1 for gt in ground_truth_labels if gt == 0)
    weight_positive = n_negative / (n_positive + n_negative) if n_positive > 0 else 1.0
    weight_negative = n_positive / (n_positive + n_negative) if n_negative > 0 else 1.0
    weights = [weight_positive if gt == 1 else weight_negative for gt in ground_truth_labels]

    accuracy = accuracy_score(ground_truth_labels, predicted_labels)
    precision = precision_score(ground_truth_labels, predicted_labels, zero_division=0)
    recall = recall_score(ground_truth_labels, predicted_labels, zero_division=0)
    f1 = f1_score(ground_truth_labels, predicted_labels, zero_division=0, sample_weight=weights)
    try:
        fpr, tpr, _ = roc_curve(ground_truth_labels, normalized_scores, sample_weight=weights)
        roc_auc = auc(fpr, tpr)
    except ValueError:
        roc_auc = None
        print("Warning: AUC-ROC could not be computed due to label imbalance.")

    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC-ROC: {roc_auc if roc_auc is not None else 'nan'}")
    return accuracy, precision, recall, f1, roc_auc

# --------------------------------------------
# 🧱 Step 5: 3D Thermal Point Cloud
# --------------------------------------------
def create_thermal_point_cloud(gray_frame, scale=0.15, downsample=2):
    try:
        h, w = gray_frame.shape
        y_coords, x_coords = np.mgrid[0:h:downsample, 0:w:downsample]
        z_values = gray_frame[y_coords, x_coords] * scale
        points = np.stack([x_coords.flatten(), y_coords.flatten(), z_values.flatten()], axis=1)
        normalized_intensity = gray_frame[y_coords, x_coords] / 255.0
        colors = plt.cm.inferno(normalized_intensity)[:, :, :3].reshape(-1, 3)
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points)
        pc.colors = o3d.utility.Vector3dVector(colors)
        return pc
    except Exception as e:
        print(f"Error creating point cloud: {str(e)}")
        return None

def analyze_point_cloud(frames, normalized_scores, output_ply="thermal_point_cloud_clustered.ply"):
    print("Analyzing thermal point cloud...")
    check_memory()
    top_idx = np.argmax(normalized_scores)
    pc = create_thermal_point_cloud(frames[top_idx])
    if pc is None:
        return None

    points = np.asarray(pc.points)
    intensity = points[:, 2]
    intensity_threshold = np.percentile(intensity, 90)
    high_intensity = intensity > intensity_threshold

    if np.sum(high_intensity) > 10:
        clustering = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=3, cluster_selection_epsilon=0.5)
        clustering.fit(points[high_intensity])
        labels = clustering.labels_
        n_clusters = len(np.unique(labels)) - (1 if -1 in labels else 0)
        print(f"Found {n_clusters} hotspot clusters in frame {top_idx}")

        colors = np.asarray(pc.colors)
        cluster_colors = np.zeros_like(colors)
        cluster_colors[~high_intensity] = colors[~high_intensity]
        for label in np.unique(labels):
            mask = (labels == label)
            cluster_colors[high_intensity][mask] = [0.5, 0.5, 0.5] if label == -1 else plt.cm.tab10(label % 10)[:3]
        pc.colors = o3d.utility.Vector3dVector(cluster_colors)

    o3d.io.write_point_cloud(output_ply, pc, write_ascii=False, compressed=True)
    print(f"✅ Saved point cloud as {output_ply}")
    check_memory()
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return pc

# --------------------------------------------
# 📸 Step 6: Grad-CAM Visualization
# --------------------------------------------
def optimized_grad_cam(model, processor, frame, text_prompts, frame_idx, device="cuda"):
    try:
        check_memory()
        resized_frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
        img = Image.fromarray(frame)
        inputs = processor(text=text_prompts, images=img, return_tensors="pt", padding=True).to(device)
        img_tensor = inputs['pixel_values'].requires_grad_(True)

        with torch.amp.autocast('cuda', enabled=use_amp and device == "cuda"):
            outputs = model(pixel_values=img_tensor, input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
            score = outputs.logits_per_image.softmax(dim=1)[0][0]

        model.zero_grad()
        score.backward()
        grads = img_tensor.grad.cpu().numpy()
        weights = plt.cm.inferno(resized_frame / 255.0)[..., :3].mean(axis=2)
        cam = np.abs(grads * weights[np.newaxis, np.newaxis, :, :]).sum(axis=1).squeeze()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        heatmap = cv2.resize(cam, (frame.shape[1], frame.shape[0]))
        sigma = max(frame.shape[0], frame.shape[1]) / 25
        heatmap = cv2.GaussianBlur(heatmap, (11, 11), sigma)
        threshold = np.percentile(heatmap, 90)
        heatmap_binary = (heatmap > threshold).astype(np.uint8) * 255
        kernel = np.ones((5, 5), np.uint8)
        heatmap_binary = cv2.dilate(heatmap_binary, kernel, iterations=2)
        heatmap = (heatmap * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(cv2.applyColorMap(frame, cv2.COLORMAP_INFERNO), 0.5, heatmap_color, 0.5, 0)

        pc = create_thermal_point_cloud(frame, downsample=1)
        if pc is None:
            return overlay, 0

        points = np.asarray(pc.points)
        intensity = points[:, 2]
        intensity_threshold = np.percentile(intensity, 90)
        high_intensity = intensity > intensity_threshold
        height, width = frame.shape
        high_intensity_mask = np.zeros((height, width), dtype=np.uint8)
        for y, x, _ in points[high_intensity]:
            if 0 <= int(y) < height and 0 <= int(x) < width:
                high_intensity_mask[int(y), int(x)] = 255
        high_intensity_mask = cv2.dilate(high_intensity_mask, kernel, iterations=1)

        intersection = np.logical_and(high_intensity_mask, heatmap_binary).sum()
        union = np.logical_or(high_intensity_mask, heatmap_binary).sum()
        iou = intersection / union if union > 0 else 0
        return overlay, iou
    except Exception as e:
        print(f"Error generating Grad-CAM for frame {frame_idx}: {str(e)}")
        return None, 0
    finally:
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

def save_heatmap_animation(frames, indices, normalized_scores, grad_cam_frames, output_video="hotspot_animation.mp4", fps=10):
    print("Creating animation...")
    check_memory()
    height, width = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    text_color = (255, 255, 255)
    text_thickness = 2

    for idx in tqdm(indices):
        color_frame = grad_cam_frames[idx][0] if idx in grad_cam_frames else cv2.applyColorMap(frames[idx], cv2.COLORMAP_INFERNO)
        if idx in grad_cam_frames:
            cv2.putText(color_frame, "Grad-CAM", (10, 120), font, font_scale, text_color, text_thickness)
        cv2.putText(color_frame, f"Frame {idx}, Score: {normalized_scores[idx]:.3f}", (10, 30), font, font_scale, text_color, text_thickness)
        cv2.putText(color_frame, "Hotspot" if normalized_scores[idx] > 0.5 else "Normal", (10, 60), font, font_scale, text_color, text_thickness)
        out.write(color_frame)

    out.release()
    print(f"✅ Animation saved as {output_video}")
    check_memory()
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return output_video

# --------------------------------------------
# 🚀 Main Pipeline
# --------------------------------------------
def main(video_path, ground_truth_csv=None):
    print("🚀 Starting thermal hotspot detection pipeline...")
    output_csv = "ground_truth_labels.csv"
    frames, fps = load_drone_video(video_path)
    raw_clip_scores, normalized_scores = process_with_clip(frames)

    # Generate or load ground truth labels
    if ground_truth_csv and os.path.exists(ground_truth_csv):
        df = pd.read_csv(ground_truth_csv)
        if 'frame_index' in df.columns and 'label' in df.columns:
            ground_truth_labels = [0] * len(frames)
            for _, row in df.iterrows():
                if row['frame_index'] < len(frames):
                    ground_truth_labels[int(row['frame_index'])] = int(row['label'])
        else:
            print("Warning: CSV must have 'frame_index' and 'label' columns. Generating new labels.")
            ground_truth_labels = None
    else:
        print("No ground truth CSV provided. Generating initial labels...")
        df = generate_initial_labels(frames, normalized_scores, output_csv)
        ground_truth_labels = [0] * len(frames)
        for _, row in df.iterrows():
            if row['frame_index'] < len(frames):
                ground_truth_labels[int(row['frame_index'])] = int(row['label'])

    # Initialize model and processor for Grad-CAM
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    # Optimize threshold and compute metrics
    threshold, predicted_labels = optimize_threshold(normalized_scores, ground_truth_labels)
    final_predicted_labels = []
    for i in range(len(predicted_labels)):
        start = max(0, i - 2)
        end = min(len(predicted_labels), i + 3)
        window = predicted_labels[start:end]
        final_predicted_labels.append(1 if sum(window) > len(window) // 2 else 0)
    predicted_labels = final_predicted_labels
    accuracy, precision, recall, f1, roc_auc = compute_metrics(ground_truth_labels, predicted_labels, normalized_scores)

    # Analyze point cloud
    pc = analyze_point_cloud(frames, normalized_scores)

    # Generate Grad-CAM for top hotspots
    hotspot_indices = [i for i, pred in enumerate(predicted_labels) if pred == 1]
    grad_cam_frames = {}
    if hotspot_indices:
        hotspot_data = [(i, normalized_scores[i]) for i in hotspot_indices]
        hotspot_data.sort(key=lambda x: x[1], reverse=True)
        top_hotspots = [idx for idx, _ in hotspot_data[:3]]

        print("\n🔥 Generating Grad-CAM visualizations (sequential)...")
        for idx in tqdm(top_hotspots, desc="Processing Grad-CAM"):
            grad_cam_img, iou = optimized_grad_cam(model, processor, frames[idx],
                                                   ["intense thermal anomaly in concrete curing", "normal concrete curing temperature profile"], idx, device)
            if grad_cam_img is not None:
                grad_cam_frames[idx] = (grad_cam_img, iou)
                cv2.imwrite(f"grad_cam_hotspot_frame_{idx}.png", grad_cam_img)
                print(f"✅ Generated Grad-CAM for frame {idx} with IoU: {iou:.4f}")

        if grad_cam_frames:
            avg_iou = np.mean([iou for _, iou in grad_cam_frames.values()])
            print(f"Generated Grad-CAM for {len(grad_cam_frames)} frames with average IoU: {avg_iou:.4f}")

    # Create animation
    if hotspot_indices:
        print(f"\n🎬 Creating animation for {len(hotspot_indices)} hotspot frames...")
        save_heatmap_animation(frames, hotspot_indices, normalized_scores, grad_cam_frames, fps=fps)

    # Clean up
    del model, processor
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    check_memory()

    # Print final label distribution
    final_df = pd.read_csv(output_csv)
    print(f"Final label distribution: {final_df['label'].value_counts().to_dict()}")

    # Return results
    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auc_roc": roc_auc,
        "hotspot_count": len(hotspot_indices),
        "threshold": threshold,
        "misclassified_count": sum(1 for gt, pred in zip(ground_truth_labels, predicted_labels) if gt != pred)
    }
    print("\n📊 Final Results Summary:")
    for key, value in results.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) and value is not None else f"  {key}: {value}")
    return results

if __name__ == "__main__":
    video_path = "/content/org_32d8ca1cb4188baf_1739865698000.mp4"
    ground_truth_csv = None
    results = main(video_path, ground_truth_csv)
