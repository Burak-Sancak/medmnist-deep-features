import numpy as np
import hashlib
import torch
import torch.nn as nn
from torchvision import models, transforms
from scipy.ndimage import rotate
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight
from scipy.stats import loguniform
import medmnist
from medmnist import INFO
import pandas as pd
from tabulate import tabulate


def tune_svm(X_train_scaled, y_train, subsample_size=1500):
    rng = np.random.RandomState(42)
    subsample_idx = rng.choice(len(X_train_scaled), size=min(subsample_size, len(X_train_scaled)), replace=False)
    X_sub = X_train_scaled[subsample_idx]
    y_sub = y_train[subsample_idx]

    param_dist = {
        "C": loguniform(0.1, 20),
        "gamma": loguniform(0.0001, 0.05),
        "kernel": ["rbf"],
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        SVC(probability=True, random_state=42, decision_function_shape="ovr", max_iter=3000, class_weight="balanced"),
        param_distributions=param_dist,
        n_iter=8,
        cv=cv,
        scoring="accuracy",
        n_jobs=1,
        verbose=1,
        random_state=42,
    )
    search.fit(X_sub, y_sub)
    print("En iyi SVM parametreleri:", search.best_params_)
    return search.best_estimator_


def evaluate_classifiers(extractor_name, dataset_name, X_train_scaled, y_train, X_test_scaled, y_test, results, class_names):
    tuned_svm = tune_svm(X_train_scaled, y_train)

    classifiers = {
        "SVM (tuned, subsample)": tuned_svm,
        "k-NN": KNeighborsClassifier(n_neighbors=5),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced", n_jobs=1),
        "XGBoost": XGBClassifier(eval_metric="mlogloss", random_state=42),
    }

    for name, clf in classifiers.items():
        if name == "XGBoost":
            sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
            clf.fit(X_train_scaled, y_train, sample_weight=sample_weights)
        elif name != "SVM (tuned, subsample)":
            clf.fit(X_train_scaled, y_train)

        preds = clf.predict(X_test_scaled)
        probs = clf.predict_proba(X_test_scaled)

        acc = accuracy_score(y_test, preds)
        f1_macro = f1_score(y_test, preds, average="macro")
        auc_macro = roc_auc_score(y_test, probs, multi_class="ovr", average="macro")

        results.append({
            "Veri Seti": dataset_name,
            "Öznitelik Çıkarıcı": extractor_name,
            "Model": name,
            "Test Accuracy": round(acc, 4),
            "Test F1 (macro)": round(f1_macro, 4),
            "Test AUC (macro)": round(auc_macro, 4),
        })
        print(f"{extractor_name} + {name}: Acc={acc:.4f} | F1={f1_macro:.4f} | AUC={auc_macro:.4f}")

        if name == "XGBoost":
            print(f"\n--- {extractor_name} + {name} sınıf bazında rapor ---")
            print(classification_report(y_test, preds, target_names=class_names, zero_division=0))


def main():
    np.random.seed(42)

    # 1. Veriyi yükle
    info = INFO["pathmnist"]
    DataClass = getattr(medmnist, info["python_class"])
    class_names = [info["label"][str(i)] for i in range(len(info["label"]))]

    train_dataset = DataClass(split="train", download=True)
    val_dataset   = DataClass(split="val", download=True)
    test_dataset  = DataClass(split="test", download=True)

    print("=== Sınıf dağılımı ===")
    for name, ds in [("train", train_dataset), ("val", val_dataset), ("test", test_dataset)]:
        labels = ds.labels.flatten()
        unique, counts = np.unique(labels, return_counts=True)
        print(f"{name}: toplam {len(labels)} örnek")
        for u, c in zip(unique, counts):
            print(f"  Sınıf {u} ({info['label'][str(u)]}): {c} örnek ({c/len(labels)*100:.1f}%)")

    # 2. Duplicate kontrolü ve temizleme
    def hash_images(imgs):
        hashes = {}
        for i, img in enumerate(imgs):
            h = hashlib.md5(img.tobytes()).hexdigest()
            hashes.setdefault(h, []).append(i)
        return hashes

    train_hashes = hash_images(train_dataset.imgs)
    test_hashes  = hash_images(test_dataset.imgs)

    overlap_train_test = set(train_hashes) & set(test_hashes)
    print(f"\nTrain-Test arasında ortak görüntü: {len(overlap_train_test)}")

    overlap_indices = {train_hashes[h][0] for h in overlap_train_test}
    clean_mask = np.array([i not in overlap_indices for i in range(len(train_dataset.imgs))])
    train_imgs_clean = train_dataset.imgs[clean_mask]
    train_labels_clean = train_dataset.labels[clean_mask]
    print(f"Temizleme öncesi train: {len(train_dataset.imgs)}, sonrası: {len(train_imgs_clean)}")

    # 3. Augmentation — üst sınırlı, DermaMNIST'te kullandığımız aynı mantık
    def augment_class(imgs, labels, target_class, num_new_samples):
        class_imgs = imgs[labels.flatten() == target_class]
        augmented = []
        for i in range(num_new_samples):
            img = class_imgs[i % len(class_imgs)].astype(np.float32)
            angle = np.random.uniform(-15, 15)
            rotated = rotate(img, angle, reshape=False, mode="reflect")
            if np.random.rand() > 0.5:
                rotated = np.fliplr(rotated)
            brightness = np.random.uniform(0.85, 1.15)
            rotated = np.clip(rotated * brightness, 0, 255)
            augmented.append(rotated.astype(imgs.dtype))
        return np.array(augmented)

    labels_flat = train_labels_clean.flatten()
    unique, counts = np.unique(labels_flat, return_counts=True)
    max_count = counts.max()

    max_multiplier = 4
    target_count = min(max_count, counts.min() * max_multiplier)

    print("\n=== Augmentation (üst sınırlı) ===")
    all_imgs = [train_imgs_clean]
    all_labels = [train_labels_clean]

    for cls, cnt in zip(unique, counts):
        target = max(cnt, target_count)
        num_to_add = target - cnt
        if num_to_add > 0:
            new_imgs = augment_class(train_imgs_clean, train_labels_clean, cls, num_to_add)
            new_labels = np.full((num_to_add, 1), cls, dtype=train_labels_clean.dtype)
            all_imgs.append(new_imgs)
            all_labels.append(new_labels)
            print(f"Sınıf {cls}: {cnt} -> {target} ({num_to_add} yapay örnek eklendi)")

    train_imgs_augmented = np.concatenate(all_imgs, axis=0)
    train_labels_augmented = np.concatenate(all_labels, axis=0)
    y_train = train_labels_augmented.flatten()
    y_test = test_dataset.labels.flatten()
    print(f"Augmentation sonrası train toplam: {len(train_imgs_augmented)}")

    # 4. Cihaz ve ortak preprocessing (PathMNIST zaten RGB)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nKullanılan cihaz: {device}")

    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def extract_features(imgs, model, batch_size=64):
        features = []
        with torch.no_grad():
            for i in range(0, len(imgs), batch_size):
                batch = imgs[i:i+batch_size]
                batch_tensors = torch.stack([preprocess(img) for img in batch]).to(device)
                batch_features = model(batch_tensors)
                features.append(batch_features.cpu().numpy())
                if i % (batch_size * 50) == 0:
                    print(f"  {i}/{len(imgs)} görüntü işlendi...")
        return np.concatenate(features, axis=0)

    results = []

    # 5. ResNet-18 ile öznitelik çıkarımı ve sınıflandırma
    print("\n=== ResNet-18 ile öznitelik çıkarımı ===")
    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    resnet.fc = nn.Identity()
    resnet = resnet.to(device)
    resnet.eval()

    X_train_resnet = extract_features(train_imgs_augmented, resnet)
    X_test_resnet  = extract_features(test_dataset.imgs, resnet)
    print(f"X_train_resnet şekli: {X_train_resnet.shape}")

    scaler_resnet = StandardScaler()
    X_train_resnet_scaled = scaler_resnet.fit_transform(X_train_resnet)
    X_test_resnet_scaled  = scaler_resnet.transform(X_test_resnet)

    evaluate_classifiers("ResNet-18", "PathMNIST", X_train_resnet_scaled, y_train, X_test_resnet_scaled, y_test, results, class_names)

    # 6. ViT-B/16 ile öznitelik çıkarımı ve sınıflandırma
    print("\n=== ViT-B/16 ile öznitelik çıkarımı ===")
    vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    vit.heads = nn.Identity()
    vit = vit.to(device)
    vit.eval()

    X_train_vit = extract_features(train_imgs_augmented, vit)
    X_test_vit  = extract_features(test_dataset.imgs, vit)
    print(f"X_train_vit şekli: {X_train_vit.shape}")

    scaler_vit = StandardScaler()
    X_train_vit_scaled = scaler_vit.fit_transform(X_train_vit)
    X_test_vit_scaled  = scaler_vit.transform(X_test_vit)

    evaluate_classifiers("ViT-B/16", "PathMNIST", X_train_vit_scaled, y_train, X_test_vit_scaled, y_test, results, class_names)

    # 7. Sonuçları tabloya dök ve kaydet
    results_df = pd.DataFrame(results)
    print("\n\n")
    print(tabulate(results_df, headers="keys", tablefmt="fancy_grid", showindex=False))
    results_df.to_csv("pathmnist_sonuclari.csv", index=False)
    print("\nSonuçlar 'pathmnist_sonuclari.csv' dosyasına kaydedildi.")


if __name__ == "__main__":
    main()