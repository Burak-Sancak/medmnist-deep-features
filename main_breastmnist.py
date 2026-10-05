# Bu kod, MedMNIST veri kümesini kullanarak meme kanseri sınıflandırması için derin öznitelik çıkarımı ve klasik makine öğrenimi sınıflandırıcılarını uygular.
import medmnist
from medmnist import INFO
import numpy as np


def main():
    np.random.seed(42)  # Bundan sonra her çalıştırmada augmentation aynı "rastgele" sayıları üretecek.

    info = INFO["breastmnist"]
    DataClass = getattr(medmnist, info["python_class"])

    train_dataset = DataClass(split="train", download=True)
    val_dataset   = DataClass(split="val", download=True)
    test_dataset  = DataClass(split="test", download=True)

    for name, ds in [("train", train_dataset), ("val", val_dataset), ("test", test_dataset)]:
        labels = ds.labels.flatten()
        unique, counts = np.unique(labels, return_counts=True)
        print(f"{name}: toplam {len(labels)} örnek")
        for u, c in zip(unique, counts):
            print(f"  Sınıf {u} ({info['label'][str(u)]}): {c} örnek ({c/len(labels)*100:.1f}%)")

    import hashlib

    def hash_images(imgs):
        hashes = {}
        for i, img in enumerate(imgs):
            h = hashlib.md5(img.tobytes()).hexdigest()
            hashes.setdefault(h, []).append(i)
        return hashes

    train_hashes = hash_images(train_dataset.imgs)
    val_hashes   = hash_images(val_dataset.imgs)
    test_hashes  = hash_images(test_dataset.imgs)

    train_dupes = {h: idxs for h, idxs in train_hashes.items() if len(idxs) > 1}
    print(f"Train içinde tekrarlanan görüntü sayısı: {len(train_dupes)}")

    overlap_train_val  = set(train_hashes) & set(val_hashes)
    overlap_train_test = set(train_hashes) & set(test_hashes)
    overlap_val_test    = set(val_hashes) & set(test_hashes)

    print(f"Train-Val arasında ortak görüntü: {len(overlap_train_val)}")
    print(f"Train-Test arasında ortak görüntü: {len(overlap_train_test)}")
    print(f"Val-Test arasında ortak görüntü: {len(overlap_val_test)}")

    #### Tekrarlayan veriyi temizlemek için train setinden test seti ile çakışan görüntüleri çıkarıyoruz.
    overlap_train_test = set(train_hashes) & set(test_hashes)

    for h in overlap_train_test:
        train_idx = train_hashes[h][0]
        test_idx = test_hashes[h][0]
        print(f"Çakışan görüntü — train index: {train_idx}, test index: {test_idx}")

    overlap_indices = {train_hashes[h][0] for h in overlap_train_test}
    clean_mask = np.array([i not in overlap_indices for i in range(len(train_dataset.imgs))])

    train_imgs_clean = train_dataset.imgs[clean_mask]
    train_labels_clean = train_dataset.labels[clean_mask]

    print(f"Temizleme öncesi train: {len(train_dataset.imgs)} örnek")
    print(f"Temizleme sonrası train: {len(train_imgs_clean)} örnek")

    #### malignant sınıfına ait mevcut görüntüleri alıp, her birine rastgele -15° ile +15° arası döndürme ve
    #### %50 ihtimalle yatay çevirme uygulayarak yeni, yapay ama gerçekçi görüntüler üretiyor.
    from scipy.ndimage import rotate

    def augment_minority_class(imgs, labels, target_class, num_new_samples):
        minority_imgs = imgs[labels.flatten() == target_class]
        augmented = []
        for i in range(num_new_samples):
            img = minority_imgs[i % len(minority_imgs)]
            angle = np.random.uniform(-15, 15)
            rotated = rotate(img, angle, reshape=False, mode="reflect")
            if np.random.rand() > 0.5:
                rotated = np.fliplr(rotated)
            augmented.append(rotated)
        return np.array(augmented)

    num_malignant = (train_labels_clean.flatten() == 0).sum()
    num_benign = (train_labels_clean.flatten() == 1).sum()
    num_to_add = num_benign - num_malignant

    print(f"Mevcut malignant: {num_malignant}, benign: {num_benign}")
    print(f"Eklenecek yapay malignant örnek: {num_to_add}")

    new_malignant_imgs = augment_minority_class(train_imgs_clean, train_labels_clean, target_class=0, num_new_samples=num_to_add)
    new_malignant_labels = np.zeros((num_to_add, 1), dtype=train_labels_clean.dtype)

    train_imgs_augmented = np.concatenate([train_imgs_clean, new_malignant_imgs], axis=0)
    train_labels_augmented = np.concatenate([train_labels_clean, new_malignant_labels], axis=0)

    print(f"Augmentation sonrası train: {len(train_imgs_augmented)} örnek")
    unique, counts = np.unique(train_labels_augmented.flatten(), return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  Sınıf {u}: {c} örnek")

    # Derin öznitelik çıkarımı için ResNet18 modelini kullanıyoruz.
    # Modelin son katmanını kaldırıp, 512 boyutlu öznitelik vektörleri elde ediyoruz.
    import torch
    import torch.nn as nn
    from torchvision import models, transforms

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanılan cihaz: {device}")

    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    resnet.fc = nn.Identity()
    resnet = resnet.to(device)
    resnet.eval()

    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def extract_resnet_features(imgs, batch_size=32):
        features = []
        with torch.no_grad():
            for i in range(0, len(imgs), batch_size):
                batch = imgs[i:i+batch_size]
                batch_tensors = torch.stack([preprocess(img) for img in batch]).to(device)
                batch_features = resnet(batch_tensors)
                features.append(batch_features.cpu().numpy())
        return np.concatenate(features, axis=0)

    X_train = extract_resnet_features(train_imgs_augmented)
    y_train = train_labels_augmented.flatten()

    X_val = extract_resnet_features(val_dataset.imgs)
    y_val = val_dataset.labels.flatten()

    X_test = extract_resnet_features(test_dataset.imgs)
    y_test = test_dataset.labels.flatten()

    print(f"X_train şekli: {X_train.shape}")
    print(f"X_val şekli: {X_val.shape}")
    print(f"X_test şekli: {X_test.shape}")

    # Çıkardığımız 512 boyutlu derin öznitelikleri klasik ML sınıflandırıcılarına veriyoruz.
    # SVM, k-NN, Random Forest ve XGBoost kullanarak sınıflandırma yapıyoruz ve sonuçları karşılaştırıyoruz.
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    classifiers = {
        "SVM": SVC(kernel="rbf", probability=True, random_state=42),
        "k-NN": KNeighborsClassifier(n_neighbors=5),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42),
    }

    results = []
    for name, clf in classifiers.items():
        clf.fit(X_train_scaled, y_train)
        val_preds = clf.predict(X_val_scaled)
        test_preds = clf.predict(X_test_scaled)
        test_probs = clf.predict_proba(X_test_scaled)[:, 1]

        val_acc = accuracy_score(y_val, val_preds)
        test_acc = accuracy_score(y_test, test_preds)
        test_f1 = f1_score(y_test, test_preds)
        test_auc = roc_auc_score(y_test, test_probs)

        results.append({
            "Öznitelik Çıkarıcı": "ResNet-18",
            "Model": name,
            "Val Accuracy": round(val_acc, 4),
            "Test Accuracy": round(test_acc, 4),
            "Test F1": round(test_f1, 4),
            "Test AUC": round(test_auc, 4),
        })
        print(f"{name}: Val Acc={val_acc:.4f} | Test Acc={test_acc:.4f} | Test F1={test_f1:.4f} | Test AUC={test_auc:.4f}")

    # ViT-B/16 modelini kullanarak öznitelik çıkarımı yapıyoruz.
    # Bu model, ResNet-18'e göre daha büyük ve daha güçlüdür, ancak daha fazla hesaplama kaynağı gerektirir.
    from torchvision.models import vit_b_16, ViT_B_16_Weights

    print("ViT-B/16 modeli indiriliyor (~330MB, ilk seferde biraz sürebilir)...")
    vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    vit.heads = nn.Identity()
    vit = vit.to(device)
    vit.eval()

    def extract_vit_features(imgs, batch_size=32):
        features = []
        with torch.no_grad():
            for i in range(0, len(imgs), batch_size):
                batch = imgs[i:i+batch_size]
                batch_tensors = torch.stack([preprocess(img) for img in batch]).to(device)
                batch_features = vit(batch_tensors)
                features.append(batch_features.cpu().numpy())
        return np.concatenate(features, axis=0)

    X_train_vit = extract_vit_features(train_imgs_augmented)
    X_val_vit   = extract_vit_features(val_dataset.imgs)
    X_test_vit  = extract_vit_features(test_dataset.imgs)

    print(f"X_train_vit şekli: {X_train_vit.shape}")

    scaler_vit = StandardScaler()
    X_train_vit_scaled = scaler_vit.fit_transform(X_train_vit)
    X_val_vit_scaled   = scaler_vit.transform(X_val_vit)
    X_test_vit_scaled  = scaler_vit.transform(X_test_vit)

    classifiers_vit = {
        "SVM": SVC(kernel="rbf", probability=True, random_state=42),
        "k-NN": KNeighborsClassifier(n_neighbors=5),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42),
    }
    # ViT-B/16 öznitelikleri ile klasik ML sınıflandırıcılarını eğitiyoruz ve sonuçları değerlendiriyoruz.
    for name, clf in classifiers_vit.items():
        clf.fit(X_train_vit_scaled, y_train)
        val_preds = clf.predict(X_val_vit_scaled)
        test_preds = clf.predict(X_test_vit_scaled)
        test_probs = clf.predict_proba(X_test_vit_scaled)[:, 1]

        val_acc = accuracy_score(y_val, val_preds)
        test_acc = accuracy_score(y_test, test_preds)
        test_f1 = f1_score(y_test, test_preds)
        test_auc = roc_auc_score(y_test, test_probs)

        results.append({
            "Öznitelik Çıkarıcı": "ViT-B/16",
            "Model": name,
            "Val Accuracy": round(val_acc, 4),
            "Test Accuracy": round(test_acc, 4),
            "Test F1": round(test_f1, 4),
            "Test AUC": round(test_auc, 4),
        })
        print(f"ViT + {name}: Val Acc={val_acc:.4f} | Test Acc={test_acc:.4f} | Test F1={test_f1:.4f} | Test AUC={test_auc:.4f}")

    # SVM modelini ViT-B/16 öznitelikleri ile kullanarak hiperparametre optimizasyonu yapıyoruz.
    from sklearn.model_selection import GridSearchCV, StratifiedKFold

    param_grid = {
        "C": [1, 5, 10, 20, 30],
        "gamma": ["scale", "auto", 0.0001, 0.0005, 0.001, 0.005],
        "kernel": ["rbf"],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        SVC(probability=True, random_state=42),
        param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
    )
    # GridSearchCV ile SVM modelini ViT-B/16 öznitelikleri üzerinde eğitiyoruz ve en iyi hiperparametreleri buluyoruz.
    grid_search.fit(X_train_vit_scaled, y_train)

    print("En iyi parametreler:", grid_search.best_params_)
    print("En iyi CV doğruluğu:", grid_search.best_score_)

    best_svm = grid_search.best_estimator_
    tuned_preds = best_svm.predict(X_test_vit_scaled)
    tuned_probs = best_svm.predict_proba(X_test_vit_scaled)[:, 1]

    tuned_acc = accuracy_score(y_test, tuned_preds)
    tuned_f1 = f1_score(y_test, tuned_preds)
    tuned_auc = roc_auc_score(y_test, tuned_probs)

    print("Test Accuracy:", tuned_acc)
    print("Test F1:", tuned_f1)
    print("Test AUC:", tuned_auc)

    results.append({
        "Öznitelik Çıkarıcı": "ViT-B/16 (GridSearch optimize)",
        "Model": f"SVM (C={grid_search.best_params_['C']}, gamma={grid_search.best_params_['gamma']})",
        "Val Accuracy": None,
        "Test Accuracy": round(tuned_acc, 4),
        "Test F1": round(tuned_f1, 4),
        "Test AUC": round(tuned_auc, 4),
    })

    # Sonuçları tablo halinde yazdırıyoruz ve CSV dosyasına kaydediyoruz.
    import pandas as pd
    from tabulate import tabulate

    print("\n\n\n")
    results_df = pd.DataFrame(results)
    print(tabulate(results_df, headers="keys", tablefmt="fancy_grid", showindex=False))
    results_df.to_csv("breastmnist_sonuclari.csv", index=False)
    print("\nSonuçlar 'breastmnist_sonuclari.csv' dosyasına kaydedildi.")


if __name__ == "__main__":
    main()