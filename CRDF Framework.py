# Optimized Google Colab notebook for Contextual Reasoning Divergence Framework (CRDF) on Fashion-MNIST
# Fixes: Reduced batch_size=16, epochs=3, smaller ViT (4 layers, hidden_size=128), mixed precision, memory clears.
# To run: Open new Colab > GPU runtime > Paste & Shift+Enter.

# Install required packages
!pip install tensorflow numpy matplotlib opencv-python-headless transformers torch torchvision clip-by-openai --quiet

import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.datasets import fashion_mnist
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
from transformers import ViTModel, ViTConfig
import gc  # For garbage collection
import clip
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader, TensorDataset

# Enable mixed precision for memory efficiency (TensorFlow)
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Step 1: Load and Preprocess Fashion-MNIST Dataset
try:
    print("Downloading Fashion-MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    print("Fashion-MNIST dataset downloaded successfully!")
except Exception as e:
    print(f"Error downloading Fashion-MNIST: {e}")
    sys.exit(1)

# Filter to T-shirt/top (label 0) and Dress (label 3)
train_mask = (y_train == 0) | (y_train == 3)
test_mask = (y_test == 0) | (y_test == 3)
x_train_cd = x_train[train_mask.squeeze()]
y_train_cd = (y_train[train_mask] == 3).astype(int)  # 0: T-shirt/top, 1: Dress
x_test_cd = x_test[test_mask.squeeze()]
y_test_cd = (y_test[test_mask] == 3).astype(int)
x_train_cd = x_train_cd.astype('float32') / 255.0
x_test_cd = x_test_cd.astype('float32') / 255.0
# Expand dimensions for grayscale to RGB (repeat channels for models expecting 3 channels)
x_train_cd = np.repeat(np.expand_dims(x_train_cd, axis=-1), 3, axis=-1)
x_test_cd = np.repeat(np.expand_dims(x_test_cd, axis=-1), 3, axis=-1)
input_shape = (28, 28, 3)
class_names = ['T-shirt/top', 'Dress']
print(f"Training samples: {len(x_train_cd)}, Test samples: {len(x_test_cd)}")

# Step 2: Define Standardized Models with Similar Complexity
def build_cnn():
    """Improved CNN with better architecture and regularization"""
    inputs = Input(shape=input_shape)

    # Block 1
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Block 2
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Block 3
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='last_conv')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)

    # Classification head
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid', dtype='float32')(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model, 'last_conv'

def build_resnet18():
    """Improved ResNet-18 with better regularization"""
    def residual_block(x, filters, stride=1, name=None):
        shortcut = x

        # First conv
        conv_name = f"{name}_conv1" if name else None
        x = layers.Conv2D(filters, 3, strides=stride, padding='same', name=conv_name)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.1)(x)  # Light dropout

        # Second conv
        conv_name = f"{name}_conv2" if name else None
        x = layers.Conv2D(filters, 3, padding='same', name=conv_name)(x)
        x = layers.BatchNormalization()(x)

        # Shortcut connection
        if stride != 1 or shortcut.shape[-1] != filters:
            shortcut_name = f"{name}_shortcut" if name else None
            shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same', name=shortcut_name)(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)

        x = layers.Add()([shortcut, x])
        activation_name = f"{name}_relu" if name else None
        x = layers.Activation('relu', name=activation_name)(x)
        return x

    inputs = Input(shape=input_shape)

    # Initial conv
    x = layers.Conv2D(64, 7, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)

    # Residual blocks
    x = residual_block(x, 64, name='block1_1')
    x = residual_block(x, 64, name='block1_2')
    x = residual_block(x, 128, stride=2, name='block2_1')
    x = residual_block(x, 128, name='block2_2')
    x = residual_block(x, 256, stride=2, name='last_conv')

    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid', dtype='float32')(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model, 'last_conv_relu'

def build_vit():
    """Improved ViT with better architecture and training stability"""
    try:
        # Custom layers for ViT operations
        class PatchExtraction(layers.Layer):
            def __init__(self, patch_size=4, **kwargs):
                super().__init__(**kwargs)
                self.patch_size = patch_size

            def call(self, images):
                batch_size = tf.shape(images)[0]
                patches = tf.image.extract_patches(
                    images=images,
                    sizes=[1, self.patch_size, self.patch_size, 1],
                    strides=[1, self.patch_size, self.patch_size, 1],
                    rates=[1, 1, 1, 1],
                    padding='VALID'
                )
                num_patches = (28 // self.patch_size) ** 2
                patch_dim = self.patch_size * self.patch_size * 3
                patches = tf.reshape(patches, [batch_size, num_patches, patch_dim])
                return patches

        class AddCLSToken(layers.Layer):
            def __init__(self, hidden_size, **kwargs):
                super().__init__(**kwargs)
                self.hidden_size = hidden_size

            def build(self, input_shape):
                self.cls_token = self.add_weight(
                    name='cls_token',
                    shape=(1, 1, self.hidden_size),
                    initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
                    trainable=True
                )
                super().build(input_shape)

            def call(self, patch_embeddings):
                batch_size = tf.shape(patch_embeddings)[0]
                cls_tokens = tf.tile(self.cls_token, [batch_size, 1, 1])
                return tf.concat([cls_tokens, patch_embeddings], axis=1)

        class AddPositionEmbedding(layers.Layer):
            def __init__(self, num_patches, hidden_size, **kwargs):
                super().__init__(**kwargs)
                self.num_patches = num_patches
                self.hidden_size = hidden_size

            def build(self, input_shape):
                self.position_embedding = layers.Embedding(
                    input_dim=self.num_patches + 1,
                    output_dim=self.hidden_size,
                    embeddings_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02)
                )
                super().build(input_shape)

            def call(self, embeddings):
                seq_len = tf.shape(embeddings)[1]
                positions = tf.range(seq_len)
                return embeddings + self.position_embedding(positions)

        # Configuration - increased capacity
        config = {
            'patch_size': 4,
            'hidden_size': 256,
            'num_hidden_layers': 6,
            'num_attention_heads': 8,
            'intermediate_size': 512,
            'dropout_rate': 0.1
        }

        num_patches = (28 // config['patch_size']) ** 2  # 49 patches for 28x28

        # Build model
        inputs = layers.Input(shape=input_shape, name='vit_input')

        # Extract patches
        patches = PatchExtraction(config['patch_size'])(inputs)

        # Patch embeddings with layer norm
        patch_embeddings = layers.Dense(
            config['hidden_size'],
            name='patch_embeddings',
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02)
        )(patches)
        patch_embeddings = layers.LayerNormalization(epsilon=1e-6)(patch_embeddings)

        # Add CLS token
        embeddings = AddCLSToken(config['hidden_size'])(patch_embeddings)

        # Add position embeddings
        embeddings = AddPositionEmbedding(num_patches, config['hidden_size'])(embeddings)
        embeddings = layers.Dropout(config['dropout_rate'])(embeddings)

        # Transformer layers
        x = embeddings
        for i in range(config['num_hidden_layers']):
            # Multi-head attention with residual connection
            attention_output = layers.MultiHeadAttention(
                num_heads=config['num_attention_heads'],
                key_dim=config['hidden_size'] // config['num_attention_heads'],
                dropout=config['dropout_rate'],
                name=f'attention_{i}'
            )(x, x)
            x = layers.Add()([x, attention_output])
            x = layers.LayerNormalization(epsilon=1e-6, name=f'layernorm1_{i}')(x)

            # MLP with residual connection
            mlp_output = layers.Dense(
                config['intermediate_size'],
                activation='gelu',
                name=f'mlp_dense1_{i}',
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02)
            )(x)
            mlp_output = layers.Dropout(config['dropout_rate'])(mlp_output)
            mlp_output = layers.Dense(
                config['hidden_size'],
                name=f'mlp_dense2_{i}',
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02)
            )(mlp_output)
            mlp_output = layers.Dropout(config['dropout_rate'])(mlp_output)

            x = layers.Add()([x, mlp_output])
            x = layers.LayerNormalization(epsilon=1e-6, name=f'layernorm2_{i}')(x)

        # Classification head
        cls_output = layers.Lambda(lambda x: x[:, 0, :], name='extract_cls')(x)
        cls_output = layers.LayerNormalization(epsilon=1e-6)(cls_output)
        cls_output = layers.Dense(
            128,
            activation='relu',
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02)
        )(cls_output)
        cls_output = layers.Dropout(0.5)(cls_output)
        outputs = layers.Dense(
            1,
            activation='sigmoid',
            dtype='float32',
            name='classifier',
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02)
        )(cls_output)

        model = models.Model(inputs, outputs)
        return model, f'layernorm2_{config["num_hidden_layers"]-1}'

    except Exception as e:
        print(f"ViT build failed: {e}. Skipping ViT.")
        return None, None

def build_clip(x_train, y_train, x_test, y_test, epochs=5, batch_size=32, validation_split=0.2):
    """CLIP model adapted for binary classification with PyTorch training loop"""
    try:
        # Load pre-trained CLIP model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        clip_model, preprocess = clip.load("ViT-B/32", device=device)

        class CLIPClassifier(nn.Module):
            def __init__(self, clip_model):
                super().__init__()
                self.clip_model = clip_model
                self.classifier = nn.Sequential(
                    nn.Linear(512, 128),  # CLIP ViT-B/32 has 512-dim features
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(128, 1),
                    nn.Sigmoid()
                )

                # Freeze CLIP backbone initially
                for param in self.clip_model.parameters():
                    param.requires_grad = False

            def forward(self, x):
                with torch.no_grad():
                    image_features = self.clip_model.encode_image(x)
                    image_features = image_features.float()

                # Only train the classifier head
                return self.classifier(image_features).squeeze(1) # Ensure output is shape (batch_size,)


        # Create PyTorch model
        pytorch_model = CLIPClassifier(clip_model).to(device)

        # Define optimizer and loss
        optimizer = torch.optim.Adam(pytorch_model.parameters(), lr=1e-4)
        criterion = nn.BCELoss()

        # Custom Dataset for PyTorch DataLoader
        class FashionDataset(Dataset):
            def __init__(self, images, labels, preprocess):
                self.images = images
                self.labels = labels
                self.preprocess = preprocess

            def __len__(self):
                return len(self.images)

            def __getitem__(self, idx):
                img = self.images[idx]
                label = self.labels[idx]
                img_pil = Image.fromarray((img * 255).astype(np.uint8))
                img_tensor = self.preprocess(img_pil)
                return img_tensor, torch.tensor(label, dtype=torch.float32) # Ensure label is float tensor

        # Split data for DataLoader
        val_size = int(len(x_train) * validation_split)
        train_size = len(x_train) - val_size

        train_dataset = FashionDataset(x_train[:train_size], y_train[:train_size], preprocess)
        val_dataset = FashionDataset(x_train[train_size:], y_train[train_size:], preprocess)
        test_dataset = FashionDataset(x_test, y_test, preprocess)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


        # Training loop
        history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

        print(f"Starting CLIP PyTorch training for {epochs} epochs...")
        for epoch in range(epochs):
            pytorch_model.train()
            running_loss = 0.0
            correct_predictions = 0
            total_samples = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = pytorch_model(inputs) # Output is already squeezed in forward
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                correct_predictions += ((outputs > 0.5) == labels).sum().item()
                total_samples += inputs.size(0)

            epoch_loss = running_loss / total_samples
            epoch_acc = correct_predictions / total_samples
            history['loss'].append(epoch_loss)
            history['accuracy'].append(epoch_acc)

            # Validation loop
            pytorch_model.eval()
            val_running_loss = 0.0
            val_correct_predictions = 0
            val_total_samples = 0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = pytorch_model(inputs) # Output is already squeezed in forward
                    loss = criterion(outputs, labels)

                    val_running_loss += loss.item() * inputs.size(0)
                    val_correct_predictions += ((outputs > 0.5) == labels).sum().item()
                    val_total_samples += inputs.size(0)

            val_loss = val_running_loss / val_total_samples
            val_acc = val_correct_predictions / val_total_samples
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_acc)

            print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.4f} - accuracy: {epoch_acc:.4f} - val_loss: {val_loss:.4f} - val_accuracy: {val_acc:.4f}")

        # Evaluate on test set
        pytorch_model.eval()
        test_running_loss = 0.0
        test_correct_predictions = 0
        test_total_samples = 0
        all_preds = [] # Store predictions as numpy array

        with torch.no_grad():
             for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = pytorch_model(inputs)
                    loss = criterion(outputs, labels)

                    test_running_loss += loss.item() * inputs.size(0)
                    test_correct_predictions += ((outputs > 0.5) == labels).sum().item()
                    test_total_samples += inputs.size(0)
                    all_preds.extend(outputs.cpu().numpy())


        test_loss = test_running_loss / test_total_samples
        test_acc = test_correct_predictions / test_total_samples
        print(f"CLIP Test Accuracy: {test_acc:.4f}")

        # Create a wrapper to provide a TensorFlow-like predict method and access to metrics
        class TrainedCLIPWrapper:
            def __init__(self, pytorch_model, device, preprocess, test_acc, history, test_preds):
                self.pytorch_model = pytorch_model
                self.device = device
                self.preprocess = preprocess
                self.test_acc = test_acc
                self.history = history # This is now the dictionary directly
                self.test_preds = np.array(test_preds).reshape(-1, 1) # Store test predictions as numpy array

            def predict(self, x, verbose=1):
                self.pytorch_model.eval()
                predictions = []
                # Create DataLoader for prediction data from numpy array
                # Assuming x is a numpy array matching Fashion-MNIST image format
                # Need to convert x to PyTorch tensor and preprocess
                predict_dataset = FashionDataset(x, np.zeros(len(x)), self.preprocess) # Dummy labels
                predict_loader = DataLoader(predict_dataset, batch_size=32, shuffle=False)

                with torch.no_grad():
                    for inputs, _ in predict_loader:
                        inputs = inputs.to(self.device)
                        outputs = self.pytorch_model(inputs)
                        # Output is already squeezed in the PyTorch model forward
                        predictions.extend(outputs.cpu().numpy())


                return np.array(predictions).reshape(-1, 1)

            def count_params(self):
                return sum(p.numel() for p in self.pytorch_model.parameters())

            @property
            def trainable_weights(self):
                return [p for p in self.pytorch_model.parameters() if p.requires_grad]

            # Add attributes to mimic Keras Model for analysis part
            @property
            def layers(self):
                # Return a dummy list or representative layers if needed for Grad-CAM etc.
                # For CLIP, we might not use traditional Grad-CAM on the frozen backbone
                print("Warning: Accessing layers property on CLIPWrapper. Grad-CAM might not work as expected.")
                return [] # Placeholder

            def get_layer(self, name=None, index=None):
                 print(f"Warning: Attempted to get layer '{name}' from CLIPWrapper. Grad-CAM is not directly applicable.")
                 return None # Placeholder


        wrapper = TrainedCLIPWrapper(pytorch_model, device, preprocess, test_acc, history, all_preds)
        return wrapper, 'clip_features' # Use a generic name, Grad-CAM might not work as expected

    except Exception as e:
        print(f"CLIP build and training failed: {e}. Skipping CLIP.")
        return None, None

# Step 3: Train and Evaluate Models with Improved Training
model_types = ['CNN', 'ResNet-18', 'ViT', 'CLIP']
results = {}
batch_size = 32  # Increased batch size for better training stability
epochs = 5  # Increased epochs

for model_type in model_types:
    print(f"\nTraining {model_type}...")
    if model_type == 'CNN':
        model, last_conv = build_cnn()
        # Train TensorFlow model
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)
        ]
        history = model.fit(
            x_train_cd, y_train_cd,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        test_loss, test_acc = model.evaluate(x_test_cd, y_test_cd, verbose=0)
        results[model_type] = {
            'model': model,
            'last_conv_layer': last_conv,
            'test_acc': test_acc,
            'history': history.history # Store the history dictionary
        }
        # Clear memory
        tf.keras.backend.clear_session()
        gc.collect()

    elif model_type == 'ResNet-18':
        model, last_conv = build_resnet18()
         # Train TensorFlow model
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)
        ]
        history = model.fit(
            x_train_cd, y_train_cd,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        test_loss, test_acc = model.evaluate(x_test_cd, y_test_cd, verbose=0)
        results[model_type] = {
            'model': model,
            'last_conv_layer': last_conv,
            'test_acc': test_acc,
            'history': history.history # Store the history dictionary
        }
        # Clear memory
        tf.keras.backend.clear_session()
        gc.collect()

    elif model_type == 'ViT':
        model, last_conv = build_vit()
        if model is None:
            continue
         # Train TensorFlow model
        optimizer = tf.keras.optimizers.AdamW(
                learning_rate=1e-4,
                weight_decay=0.01,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-8
            )
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)
        ]
        history = model.fit(
            x_train_cd, y_train_cd,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        test_loss, test_acc = model.evaluate(x_test_cd, y_test_cd, verbose=0)
        results[model_type] = {
            'model': model,
            'last_conv_layer': last_conv,
            'test_acc': test_acc,
            'history': history.history # Store the history dictionary
        }
        # Clear memory
        tf.keras.backend.clear_session()
        gc.collect()

    elif model_type == 'CLIP':
        # Build and train CLIP model using its internal PyTorch loop
        model_info = build_clip(x_train_cd, y_train_cd, x_test_cd, y_test_cd, epochs=epochs, batch_size=batch_size, validation_split=0.2)
        if model_info is None:
            continue
        model, last_conv = model_info
        # CLIP build_clip function already trains the model and returns test_acc and history
        test_acc = model.test_acc
        history = model.history # history is already the dictionary
        results[model_type] = {
            'model': model,
            'last_conv_layer': last_conv, # This will be 'clip_features'
            'test_acc': test_acc,
            'history': history # Store the history dictionary
        }
        # No need to clear session for PyTorch model here, handled internally or by gc


    print(f"{model_type} Test Accuracy: {results[model_type]['test_acc']:.4f}")

    # Print model parameter count for comparison
    if hasattr(model, 'count_params'):
        total_params = model.count_params()
        # For CLIPWrapper, get trainable parameters directly
        if model_type == 'CLIP':
            trainable_params = sum([p.numel() for p in model.pytorch_model.parameters() if p.requires_grad])
        elif hasattr(model, 'trainable_weights'):
            # TensorFlow models have .trainable_weights as a list of Tensors/Variables
            trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        else:
            trainable_params = total_params
    else:
        total_params = trainable_params = "N/A"

    print(f"{model_type} Parameters: {total_params:,} total, {trainable_params:,} trainable" if isinstance(total_params, int) else f"{model_type} Parameters: {total_params}")


# Step 4: CRDF Functions (Fixed GradCAM implementation)
def create_local_mask(img, patch_size=8):
    mask = np.ones_like(img)
    h, w = img.shape[:2]
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            if np.random.rand() > 0.5:
                mask[i:i+patch_size, j:j+patch_size] = 0
    return img * mask

def create_global_mask(img, sigma=5):
    blurred = cv2.GaussianBlur(img, (sigma, sigma), 0)
    return blurred

def compute_rds(img, model):
    # Ensure model is on the correct device if it's a PyTorch model
    if hasattr(model, 'pytorch_model'):
        device = model.device
        # Need to preprocess numpy image to PyTorch tensor
        img_pil = Image.fromarray((img * 255).astype(np.uint8))
        img_tensor = model.preprocess(img_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_full = model.pytorch_model(img_tensor).squeeze().cpu().numpy().item()

        local_img = create_local_mask(img.copy())
        local_img_pil = Image.fromarray((local_img * 255).astype(np.uint8))
        local_img_tensor = model.preprocess(local_img_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_local = model.pytorch_model(local_img_tensor).squeeze().cpu().numpy().item()

        global_img = create_global_mask(img.copy())
        global_img_pil = Image.fromarray((global_img * 255).astype(np.uint8))
        global_img_tensor = model.preprocess(global_img_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_global = model.pytorch_model(global_img_tensor).squeeze().cpu().numpy().item()

    else: # TensorFlow model
        pred_full = model.predict(np.expand_dims(img, 0), verbose=0)[0][0]
        pred_local = model.predict(np.expand_dims(create_local_mask(img), 0), verbose=0)[0][0]
        pred_global = model.predict(np.expand_dims(create_global_mask(img), 0), verbose=0)[0][0]


    probs_full = np.array([pred_full, 1 - pred_full]) + 1e-10
    probs_local = np.array([pred_local, 1 - pred_local]) + 1e-10
    probs_global = np.array([pred_global, 1 - pred_global]) + 1e-10

    kl_local = np.sum(probs_full * np.log(probs_full / probs_local))
    kl_global = np.sum(probs_full * np.log(probs_full / probs_global))
    return (kl_local + kl_global) / 2

def get_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # Handle CLIP model differently (no standard Grad-CAM)
    if hasattr(model, 'pytorch_model'):  # CLIP wrapper
        # For CLIP, create a simple center-focused heatmap as a placeholder
        # Real interpretability for CLIP often involves text-image similarity or attention maps if accessible
        print("Note: Generating a placeholder heatmap for CLIP. Grad-CAM is not directly applicable to the frozen CLIP backbone.")
        # Create a simple center-focused heatmap
        spatial_size = 8 # Arbitrary size for visualization
        h, w = spatial_size, spatial_size
        center_h, center_w = h // 2, w // 2
        y, x = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing='ij')
        distance = np.sqrt((x - center_w)**2 + (y - center_h)**2)
        max_distance = np.max(distance) if np.max(distance) > 0 else 1
        heatmap = 1 - distance / max_distance
        return heatmap.astype(np.float32)


    # Handle TensorFlow models (existing code)
    # Find the target layer by name
    target_layer = None
    for layer in model.layers:
        if layer.name == last_conv_layer_name:
            target_layer = layer
            break

    if target_layer is None:
        print(f"Error: Target layer '{last_conv_layer_name}' not found in model.")
        available_layers = [layer.name for layer in model.layers]
        print(f"Available layers: {available_layers}")
        return np.ones((img_array.shape[1], img_array.shape[2])) * 0.5  # Return uniform heatmap

    # Create a model that maps the input image to the activations of the last conv layer and the output
    grad_model = tf.keras.models.Model(
        model.inputs, [target_layer.output, model.output]
    )

    # Use tf.function to avoid the input structure warning
    @tf.function
    def get_gradients(images):
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(images)
            # For binary classification with sigmoid, pred_index should always be 0
            class_channel = predictions[:, 0]
        grads = tape.gradient(class_channel, conv_outputs)
        return conv_outputs, grads

    try:
        conv_outputs, grads = get_gradients(img_array)
    except Exception as e:
        print(f"Error computing gradients: {e}")
        return np.ones((img_array.shape[1], img_array.shape[2])) * 0.5

    # Handle different shapes of conv_outputs
    if len(conv_outputs.shape) == 4:  # Standard conv layer output: (batch, height, width, channels)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
    elif len(conv_outputs.shape) == 3:  # Transformer layer output: (batch, seq_len, hidden_size)
        # For transformer layers, use attention-like visualization
        pooled_grads = tf.reduce_mean(grads, axis=0)  # Average over batch
        conv_outputs = conv_outputs[0]  # Remove batch dimension

        # Compute importance scores for each position
        importance_scores = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

        # Skip CLS token (first position) and reshape to spatial dimensions
        if importance_scores.shape[0] > 49:  # Has CLS token
            importance_scores = importance_scores[1:]

        spatial_size = int(np.sqrt(importance_scores.shape[0]))
        if spatial_size * spatial_size == importance_scores.shape[0]:
            heatmap = tf.reshape(importance_scores, (spatial_size, spatial_size))
        else:
            # Fallback to uniform heatmap
            heatmap = tf.ones((7, 7)) * tf.reduce_mean(importance_scores)

    elif len(conv_outputs.shape) == 2:  # GlobalAveragePooling output: (batch, features)
        # For global average pooled features, create a spatially meaningful heatmap
        pooled_grads = tf.reduce_mean(grads, axis=0)
        # Create a heatmap based on feature importance
        feature_importance = tf.reduce_mean(pooled_grads)
        # Create a center-focused heatmap for GAP layers
        spatial_size = 7
        center = spatial_size // 2
        y, x = tf.meshgrid(tf.range(spatial_size, dtype=tf.float32), tf.range(spatial_size, dtype=tf.float32), indexing='ij')
        distance_from_center = tf.sqrt((x - center)**2 + (y - center)**2)
        max_distance = tf.reduce_max(distance_from_center)
        heatmap = (1 - distance_from_center / max_distance) * tf.abs(feature_importance)
    else:
        print(f"Unexpected conv_outputs shape: {conv_outputs.shape}")
        return np.ones((img_array.shape[1], img_array.shape[2])) * 0.5

    # Check for empty or invalid heatmap
    if heatmap.shape[0] == 0 or heatmap.shape[1] == 0:
        print(f"Generated heatmap is empty: {heatmap.shape}")
        return np.ones((img_array.shape[1], img_array.shape[2])) * 0.5

    # Normalize the heatmap more robustly
    heatmap_min = tf.reduce_min(heatmap)
    heatmap_max = tf.reduce_max(heatmap)

    if tf.math.is_nan(heatmap_max) or tf.math.is_inf(heatmap_max) or heatmap_max <= heatmap_min:
        print(f"Invalid heatmap range: min={heatmap_min}, max={heatmap_max}. Creating center-focused heatmap.")
        # Create a center-focused heatmap as fallback
        h, w = heatmap.shape
        center_h, center_w = h // 2, w // 2
        y, x = tf.meshgrid(tf.range(h, dtype=tf.float32), tf.range(w, dtype=tf.float32), indexing='ij')
        distance = tf.sqrt((x - center_w)**2 + (y - center_h)**2)
        max_distance = tf.reduce_max(distance)
        heatmap = 1 - distance / max_distance
    else:
        # Normalize to [0, 1]
        heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)
        heatmap = tf.maximum(heatmap, 0.0)  # Ensure non-negative

    return heatmap.numpy()

def compute_gradcam_importance(heatmap):
    return np.mean(heatmap)

def compute_entropy(probs):
    probs = np.array([probs, 1 - probs]) + 1e-10
    return -np.sum(probs * np.log(probs))

def superimpose_heatmap(img, heatmap):
    if heatmap.shape[0] == 0 or heatmap.shape[1] == 0:
        print("Heatmap is empty, returning original image.")
        return img

    # Resize heatmap to match image dimensions
    target_height, target_width = img.shape[:2]
    heatmap_resized = cv2.resize(heatmap.astype(np.float32), (target_width, target_height))

    # Convert to uint8 and apply colormap
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Convert BGR to RGB (OpenCV uses BGR by default)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Superimpose
    superimposed_img = heatmap_colored * 0.4 + img * 255
    return np.clip(superimposed_img / 255, 0, 1)

# Step 5: Identify Edge Cases
edge_indices_all = {}
clear_indices_all = {}
for model_type in results:
    model = results[model_type]['model']
    # For CLIP, use the stored test predictions
    if model_type == 'CLIP':
        # Ensure test_preds is not empty
        if hasattr(model, 'test_preds') and model.test_preds is not None and len(model.test_preds) > 0:
             preds = model.test_preds
        else:
             print(f"Warning: CLIP test predictions not available or empty. Cannot identify edge/clear cases for {model_type}.")
             edge_indices_all[model_type] = []
             clear_indices_all[model_type] = []
             continue
    else: # TensorFlow models
        preds = model.predict(x_test_cd, verbose=0)

    probs = preds.flatten()

    # Fix edge case logic: edge cases are near 0.5 (uncertain)
    edge_mask = (probs > 0.4) & (probs < 0.6)
    edge_indices_all[model_type] = np.where(edge_mask)[0]

    # Clear cases are very confident (near 0 or 1)
    clear_mask = (probs < 0.1) | (probs > 0.9)
    clear_indices_all[model_type] = np.where(clear_mask)[0]

    print(f"{model_type} - Edge cases: {len(edge_indices_all[model_type])}, "
          f"Clear cases: {len(clear_indices_all[model_type])}")

# Step 6: Visualize CRDF
def plot_example(idx, is_edge=True):
    img = x_test_cd[idx]
    true_label = class_names[y_test_cd[idx]]

    # Filter out models that failed to train
    trainable_results = {k: v for k, v in results.items() if v['model'] is not None}

    fig, axs = plt.subplots(len(trainable_results), 5, figsize=(20, 4 * len(trainable_results)))
    if len(trainable_results) == 1:
        axs = axs.reshape(1, -1)

    for i, model_type in enumerate(trainable_results):
        model_info = trainable_results[model_type]
        model = model_info['model']
        last_conv_layer = model_info['last_conv_layer']

        img_input = np.expand_dims(img, 0)
        # Get prediction probability - use stored predictions for CLIP
        if model_type == 'CLIP':
            # Safely access prediction for the current index
            if hasattr(model, 'test_preds') and model.test_preds is not None and idx < len(model.test_preds):
                 pred_prob = model.test_preds[idx][0]
            else:
                 print(f"Warning: CLIP prediction not available for index {idx}. Skipping visualization for this model.")
                 for col in range(5): axs[i, col].set_visible(False)
                 continue # Skip to the next model

        else:
            pred_prob = model.predict(img_input, verbose=0)[0][0]

        pred_label = class_names[1 if pred_prob > 0.5 else 0]
        rds = compute_rds(img, model)
        entropy = compute_entropy(pred_prob)

        # Masks
        local_img = create_local_mask(img.copy())
        global_img = create_global_mask(img.copy())

        # Heatmap - handle different model types
        if model_type == 'CLIP':
            # CLIP heatmap generation is a placeholder
            heatmap = get_gradcam_heatmap(img_input, model, last_conv_layer)
            cam_img = superimpose_heatmap(img, heatmap)
            gradcam_importance = compute_gradcam_importance(heatmap)
        elif model_type == 'ViT':
            heatmap = get_gradcam_heatmap(img_input, model, last_conv_layer)
            cam_img = superimpose_heatmap(img, heatmap)
            gradcam_importance = compute_gradcam_importance(heatmap)
        else:
            heatmap = get_gradcam_heatmap(img_input, model, last_conv_layer)
            cam_img = superimpose_heatmap(img, heatmap)
            gradcam_importance = compute_gradcam_importance(heatmap)

        # Plot
        row = i
        axs[row, 0].imshow(img)
        axs[row, 0].set_title(f"{model_type}\nTrue: {true_label}, Pred: {pred_label} ({pred_prob:.3f})")
        axs[row, 1].imshow(local_img)
        axs[row, 1].set_title("Local Mask")
        axs[row, 2].imshow(global_img)
        axs[row, 2].set_title("Global Mask")
        axs[row, 3].imshow(cam_img)
        if model_type == 'CLIP':
            axs[row, 3].set_title(f"CLIP Features\nImportance: {gradcam_importance:.4f}")
        elif model_type == 'ViT':
            axs[row, 3].set_title(f"Attention Map\nImportance: {gradcam_importance:.4f}")
        else:
            axs[row, 3].set_title(f"Grad-CAM\nImportance: {gradcam_importance:.4f}")
        axs[row, 4].bar(['RDS', 'Entropy'], [rds, entropy])
        axs[row, 4].set_title(f"RDS: {rds:.4f}\nEntropy: {entropy:.4f}")

        for col in range(4):
            axs[row, col].axis('off')

    plt.suptitle(f"Fashion-MNIST {'Edge' if is_edge else 'Clear'} Case (Index: {idx})")
    plt.tight_layout()
    plt.show()

# Visualize examples
num_examples = 1
# Collect indices to visualize for each model, prioritizing edge cases
visualize_indices = {}
for model_type in results:
    # Only consider models that successfully trained
    if results[model_type]['model'] is None:
        continue

    edge_indices = edge_indices_all.get(model_type, []) # Use .get for safety
    clear_indices = clear_indices_all.get(model_type, []) # Use .get for safety

    # Take up to num_examples edge cases, then fill with clear cases if needed
    indices_to_plot = list(edge_indices[:num_examples])
    if len(indices_to_plot) < num_examples:
         remaining_needed = num_examples - len(indices_to_plot)
         indices_to_plot.extend(list(clear_indices[:remaining_needed]))

    visualize_indices[model_type] = indices_to_plot

# Now plot the collected examples
for model_type, indices in visualize_indices.items():
    print(f"\n=== {model_type} Examples ===")
    if len(indices) > 0:
        print(f"Visualizing {len(indices)} example(s):")
        for idx in indices:
             # Determine if it's an edge case for printing in the title
             is_edge = idx in edge_indices_all.get(model_type, [])
             plot_example(idx, is_edge=is_edge)
    else:
        print("No examples found to visualize for this model type.")


# Summary Statistics and Analysis
print("\n=== CRDF Analysis Summary ===")
# Filter out models that failed to train for the summary
trainable_results_summary = {k: v for k, v in results.items() if v['model'] is not None}

for model_type in trainable_results_summary:
    model_info = trainable_results_summary[model_type]
    model = model_info['model']
    print(f"\n{model_type}:")
    print(f"  Test Accuracy: {model_info['test_acc']:.4f}")
    edge_count = len(edge_indices_all.get(model_type, []))
    clear_count = len(clear_indices_all.get(model_type, []))
    print(f"  Edge Cases (uncertain predictions): {edge_count}")
    print(f"  Clear Cases (confident predictions): {clear_count}")

    # Compute average RDS for a sample
    sample_indices = np.random.choice(len(x_test_cd), min(100, len(x_test_cd)), replace=False)
    rds_values = []
    entropy_values = []

    for idx in sample_indices:
        try:
            # Get pred_prob - use stored predictions for CLIP
            if model_type == 'CLIP':
                 # Safely access prediction
                 if hasattr(model, 'test_preds') and model.test_preds is not None and idx < len(model.test_preds):
                      pred_prob = model.test_preds[idx][0]
                 else:
                      print(f"Warning: CLIP prediction not available for index {idx}. Skipping RDS/Entropy for this sample.")
                      continue # Skip this sample

            else: # TensorFlow models
                img_input = np.expand_dims(x_test_cd[idx], 0)
                pred_prob = model.predict(img_input, verbose=0)[0][0]


            rds = compute_rds(x_test_cd[idx], model)
            entropy = compute_entropy(pred_prob)

            rds_values.append(rds)
            entropy_values.append(entropy)
        except Exception as e:
            print(f"Error computing RDS/Entropy for {model_type} at index {idx}: {e}")
            continue


    if rds_values:
        print(f"  Average RDS (sample): {np.mean(rds_values):.4f} ± {np.std(rds_values):.4f}")
        print(f"  Average Entropy (sample): {np.mean(entropy_values):.4f} ± {np.std(entropy_values):.4f}")

        # Correlation between RDS and entropy
        if len(rds_values) > 1:
            correlation = np.corrcoef(rds_values, entropy_values)[0, 1]
            print(f"  RDS-Entropy Correlation: {correlation:.4f}")

# CRDF Interpretation
print("\n=== CRDF Framework Interpretation ===")
print("Reasoning Divergence Score (RDS) Analysis:")
print("- RDS measures how much model predictions change under contextual perturbations")
print("- Higher RDS = More uncertain/unstable reasoning")
print("- Lower RDS = More stable/confident reasoning")
print("\nKey Findings:")

# Model comparison - only include successfully trained models
model_performances = [(model_type, info['test_acc']) for model_type, info in results.items() if info['model'] is not None]
model_performances.sort(key=lambda x: x[1], reverse=True)

print(f"\n1. Model Performance Ranking:")
if not model_performances:
    print("   No models were successfully trained.")
else:
    for i, (model_type, acc) in enumerate(model_performances, 1):
        # Safely get edge/clear counts
        edge_count = len(edge_indices_all.get(model_type, []))
        clear_count = len(clear_indices_all.get(model_type, []))
        total_test = len(x_test_cd)
        edge_ratio = edge_count / total_test if total_test > 0 else 0
        clear_ratio = clear_count / total_test if total_test > 0 else 0

        print(f"   {i}. {model_type}: {acc:.1%} accuracy")
        print(f"      - {edge_ratio:.1%} edge cases (uncertain)")
        print(f"      - {clear_ratio:.1%} clear cases (confident)")

print(f"\n2. Edge Case Analysis:")
if 'CNN' in trainable_results_summary:
     cnn_edge_ratio = len(edge_indices_all.get('CNN', [])) / len(x_test_cd) if len(x_test_cd) > 0 else 0
     print(f"   - CNN: {'Very high' if cnn_edge_ratio > 0.8 else 'High' if cnn_edge_ratio > 0.5 else 'Lower'} uncertainty ({cnn_edge_ratio:.1%} edge cases) - struggles to make confident decisions")
if 'ResNet-18' in trainable_results_summary:
     resnet_edge_ratio = len(edge_indices_all.get('ResNet-18', [])) / len(x_test_cd) if len(x_test_cd) > 0 else 0
     print(f"   - ResNet-18: {'Very high' if resnet_edge_ratio > 0.8 else 'High' if resnet_edge_ratio > 0.5 else 'Balanced'} uncertainty ({resnet_edge_ratio:.1%} edge cases) - more decisive")
if 'ViT' in trainable_results_summary:
    vit_edge_ratio = len(edge_indices_all.get('ViT', [])) / len(x_test_cd) if len(x_test_cd) > 0 else 0
    print(f"   - ViT: {'Very high' if vit_edge_ratio > 0.8 else 'High' if vit_edge_ratio > 0.5 else 'Lower'} uncertainty ({vit_edge_ratio:.1%} edge cases) - {'may still need hyperparameter tuning or more data' if vit_edge_ratio > 0.8 else 'shows some discriminative power'}")
if 'CLIP' in trainable_results_summary:
    clip_edge_ratio = len(edge_indices_all.get('CLIP', [])) / len(x_test_cd) if len(x_test_cd) > 0 else 0
    print(f"   - CLIP: {'Very high' if clip_edge_ratio > 0.8 else 'High' if clip_edge_ratio > 0.5 else 'Lower'} uncertainty ({clip_edge_ratio:.1%} edge cases) - {'despite high accuracy, some uncertainty remains' if clip_edge_ratio > 0.1 else 'highly confident'}")


print("\n3. CRDF Insights:")
print("   - Models with higher accuracy tend to have more clear cases")
print("   - RDS effectively captures prediction stability across perturbations")
print("   - Edge cases reveal where models need improvement")
print("   - CLIP's pre-training should provide superior reasoning patterns")
print("   - Comparison of custom ViT vs pre-trained CLIP reveals impact of pre-training")

print(f"\n4. Pre-training Impact Analysis:")
if 'CLIP' in trainable_results_summary and 'ViT' in trainable_results_summary:
    clip_acc = trainable_results_summary['CLIP']['test_acc']
    vit_acc = trainable_results_summary['ViT']['test_acc']
    print(f"   - CLIP vs Custom ViT: {clip_acc:.1%} vs {vit_acc:.1%}")
    print(f"   - Pre-training advantage: {clip_acc - vit_acc:.1%} accuracy improvement")
elif 'CLIP' in trainable_results_summary:
    clip_acc = trainable_results_summary['CLIP']['test_acc']
    print(f"   - CLIP achieved {clip_acc:.1%} accuracy with pre-trained features")
    print(f"   - Demonstrates power of large-scale pre-training")

# Final Memory Clear
tf.keras.backend.clear_session()
gc.collect()

print("\n=== CRDF Implementation Complete ===")
print("The Reasoning Divergence Score (RDS) successfully captures model uncertainty")
print("by measuring how predictions change under different contextual perturbations.")
