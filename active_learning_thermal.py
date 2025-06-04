import asyncio
import platform
import cv2
import numpy as np
import torch
import open3d as o3d
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from transformers import CLIPProcessor, CLIPModel, logging
from PIL import Image
import os
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances
from tqdm.auto import tqdm
import pandas as pd
import gc
import psutil
import torch.nn.functional as F
import torch.nn as nn
from sklearn.utils import resample
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomRotation, RandomResizedCrop
from transformers.optimization import get_cosine_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN

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
    ram_usage = mem_info.rss / 1024**2  # MB
    if device == "cuda":
        gpu_mem = torch.cuda.memory_allocated() / 1024**2  # MB
        print(f"RAM: {ram_usage:.2f} MB, GPU: {gpu_mem:.2f} MB")
    else:
        print(f"RAM: {ram_usage:.2f} MB")
    return ram_usage

# --------------------------------------------
# 🎞️ Step 1: Load Drone Thermal Video
# --------------------------------------------
def load_drone_video(video_path, width=320, height=240):
    print(f"Loading drone thermal video from {video_path}...")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    for _ in tqdm(range(total_frames), desc="Extracting frames"):
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
    cap.release()
    print(f"✅ Loaded {len(frames)} thermal frames.")
    check_memory()
    return frames, fps

# --------------------------------------------
# 🖼️ Step 2: Generate Initial Labels with Clustering
# --------------------------------------------
def generate_initial_labels(frames, output_csv):
    print("Generating initial labels using CLIP embeddings and K-means clustering...")
    check_memory()

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()

    prompts = [
        "intense thermal hotspot in concrete curing",
        "normal concrete temperature",
        "cool concrete surface"
    ]
    text_inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
    embeddings = []
    clip_scores = []

    with torch.no_grad():
        for i in tqdm(range(0, len(frames), 2), desc="Computing CLIP embeddings"):
            batch_frames = frames[i:i+2]
            batch_images = [Image.fromarray(f) for f in batch_frames]
            inputs = processor(images=batch_images, return_tensors="pt", padding=True).to(device)
            image_features = model.get_image_features(**inputs).cpu().numpy()
            embeddings.extend(image_features)
            outputs = model(**inputs, **text_inputs)
            logits = outputs.logits_per_image.softmax(dim=-1).cpu().numpy()
            batch_scores = logits[:, 0]
            clip_scores.extend(batch_scores)

    embeddings = np.array(embeddings)

    # K-means clustering
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)

    # Assign labels based on mean hotspot score
    cluster_scores = []
    for cluster in [0, 1]:
        cluster_indices = np.where(cluster_labels == cluster)[0]
        cluster_mean_score = np.mean([clip_scores[i] for i in cluster_indices])
        cluster_scores.append((cluster, cluster_mean_score))
    cluster_scores.sort(key=lambda x: x[1], reverse=True)
    positive_cluster = cluster_scores[0][0]

    # Validate with intensity
    mean_intensities = [np.mean(frames[i]) for i in range(len(frames))]
    pos_cluster_indices = np.where(cluster_labels == positive_cluster)[0]
    neg_cluster_indices = np.where(cluster_labels != positive_cluster)[0]
    pos_intensity = np.mean([mean_intensities[i] for i in pos_cluster_indices])
    neg_intensity = np.mean([mean_intensities[i] for i in neg_cluster_indices])
    if pos_intensity < neg_intensity:
        print("Warning: Positive cluster has lower intensity. Swapping labels.")
        positive_cluster = 1 - positive_cluster

    final_labels = [(i, 1 if cluster_labels[i] == positive_cluster else 0) for i in range(len(frames))]

    # Ensure minimum diversity
    label_counts = pd.Series([label for _, label in final_labels]).value_counts()
    print(f"Initial label distribution: {label_counts.to_dict()}")
    if len(label_counts) < 2 or min(label_counts) < 100:
        print("Warning: Insufficient label diversity. Forcing balance by reassigning.")
        sorted_indices = np.argsort(clip_scores)
        mid_point = len(frames) // 2
        final_labels = [(i, 1 if i in sorted_indices[mid_point:] else 0) for i in range(len(frames))]

    unique_labels = {idx: label for idx, label in final_labels}
    df = pd.DataFrame([(idx, label) for idx, label in unique_labels.items()], columns=['frame_index', 'label'])
    df.to_csv(output_csv, index=False)
    print(f"✅ Saved {len(df)} labels to {output_csv}")
    label_counts = df['label'].value_counts()
    print(f"Final label distribution: {label_counts.to_dict()}")
    check_memory()

    del model, processor
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    return df, embeddings

# --------------------------------------------
# 🤖 Step 3: CLIP Model Processing
# --------------------------------------------
def process_with_clip(frames, model, processor, batch_size=2):
    print("Processing with CLIP model for classification...")
    if device == "cuda":
        torch.cuda.empty_cache()
        gc.collect()
    check_memory()
    model.eval()
    prompts = ["intense thermal hotspot in concrete curing", "normal concrete temperature"]
    text_inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
    clip_scores = []
    with torch.no_grad():
        for i in tqdm(range(0, len(frames), batch_size)):
            batch_frames = frames[i:i+batch_size]
            batch_pil_images = [Image.fromarray(f) for f in batch_frames]
            with torch.amp.autocast('cuda', enabled=use_amp):
                image_inputs = processor(images=batch_pil_images, return_tensors="pt", padding=True).to(device)
                outputs = model(**text_inputs, **image_inputs)
                logits_per_image = outputs.logits_per_image.softmax(dim=-1)
                batch_scores = logits_per_image.cpu().numpy()
                clip_scores.extend(batch_scores)
    clip_scores = np.array(clip_scores)
    normalized_scores = (clip_scores - clip_scores.min(axis=0)) / (clip_scores.max(axis=0) - clip_scores.min(axis=0) + 1e-8)
    print(f"✅ Processed {len(frames)} frames.")
    check_memory()
    return clip_scores, normalized_scores[:, 0].tolist()

# --------------------------------------------
# 📊 Step 4: Classification and Metrics
# --------------------------------------------
def optimize_threshold(normalized_scores, ground_truth_labels, validation_split=0.2):
    if ground_truth_labels is None or len(np.unique(ground_truth_labels)) < 2:
        print("Insufficient label variety. Using default threshold 0.5.")
        return 0.5, [int(score > 0.5) for score in normalized_scores]

    indices = list(range(len(normalized_scores)))
    train_idx, val_idx = train_test_split(indices, test_size=validation_split, random_state=42)
    train_scores = [normalized_scores[i] for i in train_idx]
    train_labels = [ground_truth_labels[i] for i in train_idx]
    val_scores = [normalized_scores[i] for i in val_idx]
    val_labels = [ground_truth_labels[i] for i in val_idx]

    pos_count = sum(1 for l in ground_truth_labels if l == 1)
    best_f1, best_threshold = 0, 0.5
    for threshold in np.linspace(0.4, 0.8, 100):
        predicted_labels = [int(score > threshold) for score in val_scores]
        f1 = f1_score(val_labels, predicted_labels, zero_division=0)
        hotspot_count = sum(predicted_labels)
        penalty = abs(hotspot_count - pos_count) / len(val_scores) * 0.5
        adjusted_f1 = f1 - penalty
        if adjusted_f1 > best_f1:
            best_f1, best_threshold = adjusted_f1, threshold

    predicted_labels = [int(score > best_threshold) for score in normalized_scores]
    hotspot_count = sum(predicted_labels)
    if hotspot_count > 1.1 * pos_count:
        predicted_labels = sorted(range(len(normalized_scores)), key=lambda i: normalized_scores[i], reverse=True)[:int(1.1 * pos_count)]
        predicted_labels = [1 if i in predicted_labels else 0 for i in range(len(normalized_scores))]

    print(f"Optimized threshold: {best_threshold:.4f} (F1: {f1:.4f})")
    return best_threshold, predicted_labels

def compute_metrics(ground_truth_labels, predicted_labels, normalized_scores):
    if ground_truth_labels is None or len(np.unique(ground_truth_labels)) < 2:
        print("Insufficient label variety. Computing partial metrics.")
        accuracy = accuracy_score(ground_truth_labels, predicted_labels) if ground_truth_labels is not None else None
        print(f"Accuracy: {accuracy:.4f}" if accuracy is not None else "Accuracy: nan")
        print("Label distribution:", pd.Series(ground_truth_labels).value_counts().to_dict())
        return accuracy, None, None, None, None
    accuracy = accuracy_score(ground_truth_labels, predicted_labels)
    precision = precision_score(ground_truth_labels, predicted_labels, zero_division=0)
    recall = recall_score(ground_truth_labels, predicted_labels, zero_division=0)
    f1 = f1_score(ground_truth_labels, predicted_labels, zero_division=0)
    try:
        fpr, tpr, _ = roc_curve(ground_truth_labels, normalized_scores)
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
    intensity_threshold = np.percentile(intensity, 85)
    high_intensity = intensity > intensity_threshold
    if np.sum(high_intensity) > 10:
        clustering = DBSCAN(eps=10, min_samples=5, algorithm='kd_tree', n_jobs=-1).fit(points[high_intensity])
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
    return pc

# --------------------------------------------
# 🧠 Step 6: Active Learning and Grad-CAM
# --------------------------------------------
def fine_tune_clip(model, processor, uncertain_frames, device="cuda", epochs=10, batch_size=2, accum_steps=4):
    print("Fine-tuning CLIP model with class balancing...")
    check_memory()

    if device == "cpu":
        print("Warning: Fine-tuning on CPU due to GPU unavailability. This may be slower.")

    model.vision_model.encoder.layers[-1].mlp.fc2 = nn.Sequential(
        nn.Dropout(0.3),
        model.vision_model.encoder.layers[-1].mlp.fc2
    )
    model.train()

    labels = [gt for _, _, gt in uncertain_frames]
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        print("Warning: Only one class in uncertain frames. Using unweighted loss.")
        class_weights = None
    else:
        n_positive = sum(1 for l in labels if l == 1)
        n_negative = len(labels) - n_positive
        pos_weight = n_negative / n_positive if n_positive > 0 else 1.0
        class_weights = torch.tensor([1.0, pos_weight]).to(device)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.vision_model.parameters(), "lr": 2e-6},
            {"params": model.text_model.parameters(), "lr": 1e-6}
        ],
        weight_decay=0.05
    )

    total_steps = epochs * (len(uncertain_frames) // batch_size + 1)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=total_steps // 5, num_training_steps=total_steps)

    augmentations = Compose([
        RandomHorizontalFlip(p=0.5),
        RandomRotation(15),
        RandomResizedCrop(size=(224, 224), scale=(0.75, 1.0)),
    ])

    pos_frames = [(idx, frame, gt) for idx, frame, gt in uncertain_frames if gt == 1]
    neg_frames = [(idx, frame, gt) for idx, frame, gt in uncertain_frames if gt == 0]
    min_class_size = min(len(pos_frames), len(neg_frames))

    if min_class_size == 0:
        print("Warning: One class is empty. Using random sampling to balance.")
        available_frames = pos_frames if pos_frames else neg_frames
        balanced_frames = resample(available_frames, replace=True, n_samples=len(available_frames), random_state=42)
    else:
        pos_frames = resample(pos_frames, replace=True, n_samples=min_class_size, random_state=42)
        neg_frames = resample(neg_frames, replace=False, n_samples=min_class_size, random_state=42)
        balanced_frames = pos_frames + neg_frames

    np.random.shuffle(balanced_frames)

    try:
        for epoch in range(epochs):
            epoch_loss = 0
            batch_data = []
            frame_count = 0
            accum_loss = 0
            for idx, frame, gt in balanced_frames:
                batch_data.append((frame, gt))
                frame_count += 1
                if len(batch_data) == batch_size or frame_count == len(balanced_frames):
                    batch_frames, batch_labels = zip(*batch_data)
                    batch_pil_images = [augmentations(Image.fromarray(f)) for f in batch_frames]
                    batch_texts = ["intense thermal hotspot in concrete curing" if gt == 1 else "normal concrete temperature" for gt in batch_labels]
                    inputs = processor(text=batch_texts, images=batch_pil_images, return_tensors="pt", padding=True).to(device)
                    with torch.amp.autocast('cuda', enabled=use_amp and device == "cuda"):
                        outputs = model(**inputs)
                        logits_per_image = outputs.logits_per_image
                        labels = torch.tensor([gt for gt in batch_labels], dtype=torch.long).to(device)
                        alpha = 0.25
                        gamma = 2.0
                        ce_loss = F.cross_entropy(logits_per_image, labels, weight=class_weights, reduction='none')
                        pt = torch.exp(-ce_loss)
                        focal_loss = (alpha * (1 - pt) ** gamma * ce_loss).mean() / accum_steps
                    focal_loss.backward()
                    accum_loss += focal_loss.item()
                    batch_data = []
                    if (frame_count // batch_size) % accum_steps == 0 or frame_count == len(balanced_frames):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        epoch_loss += accum_loss * accum_steps
                        accum_loss = 0
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
    except Exception as e:
        print(f"Error during fine-tuning: {str(e)}")
        return model

    model.eval()
    print("✅ Fine-tuned CLIP model.")
    check_memory()
    return model

def optimized_grad_cam(model, processor, frame, text_prompts, frame_idx, device="cuda"):
    try:
        check_memory()
        resized_frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
        img = Image.fromarray(frame)
        inputs = processor(text=text_prompts, images=img, return_tensors="pt", padding=True).to(device)
        img_tensor = inputs['pixel_values'].requires_grad_(True)

        with torch.amp.autocast('cuda', enabled=use_amp and device == "cuda"):
            outputs = model(pixel_values=img_tensor, input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
            score = outputs.logits_per_image.softmax(dim=-1)[0][0]
        model.zero_grad()
        score.backward()
        grads = img_tensor.grad.cpu().numpy()
        if np.all(grads == 0):
            print(f"Warning: Zero gradients for frame {frame_idx}. Skipping Grad-CAM.")
            return None, 0
        weights = resized_frame / 255.0
        print(f"Grad shape: {grads.shape}, Weights shape: {weights.shape}")
        cam = np.abs(grads.mean(axis=1, keepdims=True) * weights[np.newaxis, np.newaxis, :, :]).sum(axis=1).squeeze()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        heatmap = cv2.resize(cam, (frame.shape[1], frame.shape[0]))
        heatmap = cv2.GaussianBlur(heatmap, (9, 9), 0)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        threshold = np.percentile(heatmap, 75)
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
        intensity_threshold = np.percentile(intensity, 95)
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
    return output_video

# --------------------------------------------
# 🚀 Main Pipeline
# --------------------------------------------
def main(video_path, ground_truth_csv=None, active_learning_iterations=5):
    print("🚀 Starting thermal hotspot detection pipeline...")
    output_csv = "ground_truth_labels.csv"
    frames, fps = load_drone_video(video_path)

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
        df, embeddings = generate_initial_labels(frames, output_csv)
        ground_truth_labels = [0] * len(frames)
        for _, row in df.iterrows():
            if row['frame_index'] < len(frames):
                ground_truth_labels[int(row['frame_index'])] = int(row['label'])

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    original_labels = ground_truth_labels.copy() if ground_truth_labels else None
    results_history = []
    selected_embeddings = []

    for iteration in range(active_learning_iterations):
        print(f"🔄 Active Learning Iteration {iteration + 1}/{active_learning_iterations}")
        raw_clip_scores, normalized_scores = process_with_clip(frames, model, processor)

        window_size = 5
        smoothed_scores = np.convolve(normalized_scores, np.ones(window_size)/window_size, mode='same')
        normalized_scores = smoothed_scores.tolist()

        threshold, predicted_labels = optimize_threshold(normalized_scores, ground_truth_labels)
        accuracy, precision, recall, f1, roc_auc = compute_metrics(ground_truth_labels, predicted_labels, normalized_scores)
        results_history.append({
            "iteration": iteration + 1,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "auc_roc": roc_auc,
            "threshold": threshold
        })

        # Hybrid margin and diversity sampling
        probs = np.array(raw_clip_scores)
        margins = np.abs(probs[:, 0] - probs[:, 1])
        margin_indices = np.argsort(margins)[:200].tolist()  # Convert to list

        # Compute embeddings for diversity
        curr_embeddings = []
        with torch.no_grad():
            for i in tqdm(margin_indices, desc="Computing embeddings for diversity"):
                img = Image.fromarray(frames[i])
                inputs = processor(images=img, return_tensors="pt", padding=True).to(device)
                emb = model.get_image_features(**inputs).cpu().numpy()
                curr_embeddings.append(emb[0])
        curr_embeddings = np.array(curr_embeddings)

        # Select diverse samples
        uncertain_indices = []
        selected_count = 0
        predicted_pos = [i for i in margin_indices if predicted_labels[i] == 1]
        predicted_neg = [i for i in margin_indices if predicted_labels[i] == 0]
        for _ in range(100):  # 100 per class
            if predicted_pos:
                if selected_embeddings:
                    distances = cosine_distances(
                        curr_embeddings[[margin_indices.index(i) for i in predicted_pos if i in margin_indices]],
                        np.array(selected_embeddings)
                    ).min(axis=1)
                    idx = predicted_pos[np.argmax(distances)]
                else:
                    idx = predicted_pos[0]
                uncertain_indices.append(idx)
                predicted_pos.remove(idx)
                selected_count += 1
            if predicted_neg and selected_count < 200:
                if selected_embeddings:
                    distances = cosine_distances(
                        curr_embeddings[[margin_indices.index(i) for i in predicted_neg if i in margin_indices]],
                        np.array(selected_embeddings)
                    ).min(axis=1)
                    idx = predicted_neg[np.argmax(distances)]
                else:
                    idx = predicted_neg[0]
                uncertain_indices.append(idx)
                predicted_neg.remove(idx)
                selected_count += 1
            if selected_count >= 200 or (not predicted_pos and not predicted_neg):
                break

        if not uncertain_indices:
            print("No uncertain frames. Skipping fine-tuning.")
            continue

        uncertain_frames = [(i, frames[i], ground_truth_labels[i]) for i in uncertain_indices]
        print(f"Selected {len(uncertain_frames)} uncertain frames with predicted labels: "
              f"{pd.Series([pred for i in uncertain_indices for pred in predicted_labels if i == predicted_labels.index(pred)]).value_counts().to_dict()}")

        # Update selected embeddings
        selected_embeddings.extend([curr_embeddings[margin_indices.index(i)] for i in uncertain_indices if i in margin_indices])

        model = fine_tune_clip(model, processor, uncertain_frames, device=device, epochs=10, batch_size=2, accum_steps=4)

        # Update labels
        post_clip_scores, post_normalized_scores = process_with_clip(frames, model, processor)
        for i, score in enumerate(post_normalized_scores):
            if i in uncertain_indices:
                ground_truth_labels[i] = 1 if score > 0.75 else 0 if score < 0.25 else ground_truth_labels[i]

        df = pd.DataFrame({"frame_index": range(len(frames)), "label": ground_truth_labels})
        df.to_csv(output_csv, index=False)
        print(f"✅ Updated {len(df)} labels to {output_csv}")
        print(f"Label distribution: {df['label'].value_counts().to_dict()}")

    threshold, predicted_labels = optimize_threshold(normalized_scores, ground_truth_labels)
    accuracy, precision, recall, f1, roc_auc = compute_metrics(ground_truth_labels, predicted_labels, normalized_scores)
    pc = analyze_point_cloud(frames, normalized_scores)
    hotspot_indices = [i for i, pred in enumerate(predicted_labels) if pred == 1]
    grad_cam_frames = {}

    if hotspot_indices:
        hotspot_data = [(i, normalized_scores[i]) for i in hotspot_indices]
        hotspot_data.sort(key=lambda x: x[1], reverse=True)
        top_hotspots = [idx for idx, _ in hotspot_data[:min(10, len(hotspot_data))]]
        print("\n🔥 Generating Grad-CAM visualizations (sequential)...")
        for idx in tqdm(top_hotspots, desc="Processing Grad-CAM"):
            grad_cam_img, iou = optimized_grad_cam(
                model, processor, frames[idx],
                ["intense thermal hotspot in concrete curing", "normal concrete temperature"], idx, device
            )
            if grad_cam_img is not None:
                grad_cam_frames[idx] = (grad_cam_img, iou)
                cv2.imwrite(f"grad_cam_hotspot_frame_{idx}.png", grad_cam_img)
                print(f"✅ Generated Grad-CAM for frame {idx} with IoU: {iou:.4f}")
        if grad_cam_frames:
            avg_iou = np.mean([iou for _, iou in grad_cam_frames.values()])
            print(f"Generated Grad-CAM for {len(grad_cam_frames)} frames with average IoU: {avg_iou:.4f}")

    if hotspot_indices:
        print(f"\n🎬 Creating animation for {len(hotspot_indices)} hotspot frames...")
        save_heatmap_animation(frames, hotspot_indices, normalized_scores, grad_cam_frames, fps=fps)

    del model, processor
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    check_memory()

    final_df = pd.read_csv(output_csv)
    print(f"Final label distribution: {final_df['label'].value_counts().to_dict()}")
    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auc_roc": roc_auc,
        "hotspot_count": len(hotspot_indices),
        "threshold": threshold,
        "misclassified_count": sum(1 for gt, pred in zip(ground_truth_labels, predicted_labels) if gt != pred),
        "results_history": results_history
    }

    print("\n📊 Final Results Summary:")
    for key, value in results.items():
        if key != "results_history":
            print(f"  {key}: {value:.4f}" if isinstance(value, float) and value is not None else f"  {key}: {value}")
    print("\n📈 Active Learning Progress:")
    for res in results_history:
        print(f"Iteration {res['iteration']}: Accuracy: {res['accuracy']:.4f}, F1: {res['f1_score']:.4f}, AUC-ROC: {res['auc_roc'] if res['auc_roc'] is not None else 'nan'}")

    return results

if __name__ == "__main__":
    video_path = "/content/org_32d8ca1cb4188baf_1739865698000.mp4"
    ground_truth_csv = None
    results = main(video_path, ground_truth_csv, active_learning_iterations=3)
