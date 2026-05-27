import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.stats import norm


def _get_model_outputs(model, data_loader, device):
    """获取模型在所有样本上的输出（logits 或 softmax probabilities）"""
    model.eval()
    outputs_list, labels_list = [], []
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            outputs_list.append(probs)
            labels_list.append(labels.numpy())
    return np.concatenate(outputs_list), np.concatenate(labels_list)


def membership_inference_attack(target_model, train_set, test_set, device,
                                 attack_samples: int = 2000):
    """成员推断攻击 (Membership Inference Attack)

    训练一个攻击模型来区分"参与了训练"和"未参与训练"的样本。
    使用模型输出的概率向量作为特征。

    返回: {"auc": float, "accuracy": float, "tpr_low_fpr": float}
    """
    n_train = min(len(train_set), attack_samples)
    n_test = min(len(test_set), attack_samples)

    # 随机采样
    train_indices = np.random.choice(len(train_set), n_train, replace=False)
    test_indices = np.random.choice(len(test_set), n_test, replace=False)

    train_subset = Subset(train_set, train_indices)
    test_subset = Subset(test_set, test_indices)

    train_loader = DataLoader(train_subset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=128, shuffle=False)

    # 获取模型输出
    train_outputs, train_labels = _get_model_outputs(
        target_model, train_loader, device)
    test_outputs, test_labels = _get_model_outputs(
        target_model, test_loader, device)

    # 构造攻击特征
    X_train_features = np.concatenate([train_outputs, test_outputs], axis=0)
    y_train_membership = np.concatenate([
        np.ones(len(train_outputs)),    # 成员 (1)
        np.zeros(len(test_outputs)),    # 非成员 (0)
    ])

    # 打乱
    shuffle_idx = np.random.permutation(len(X_train_features))
    X_train_features = X_train_features[shuffle_idx]
    y_train_membership = y_train_membership[shuffle_idx]

    # 划分攻击训练集和测试集 (50/50)
    split = len(X_train_features) // 2
    X_attack_train, X_attack_test = X_train_features[:split], X_train_features[split:]
    y_attack_train, y_attack_test = y_train_membership[:split], y_train_membership[split:]

    # 训练攻击模型 (SVM)
    attack_model = SVC(kernel="rbf", probability=True)
    attack_model.fit(X_attack_train, y_attack_train)

    # 评估攻击模型
    y_pred_proba = attack_model.predict_proba(X_attack_test)[:, 1]
    y_pred = attack_model.predict(X_attack_test)

    auc = roc_auc_score(y_attack_test, y_pred_proba)
    acc = accuracy_score(y_attack_test, y_pred)

    # TPR @ 低 FPR (FPR <= 0.05)
    fpr_threshold = 0.05
    sorted_indices = np.argsort(y_pred_proba)[::-1]
    tpr_at_low_fpr = 0.0
    for threshold in np.linspace(0.5, 1.0, 100):
        preds = (y_pred_proba >= threshold).astype(int)
        fp = np.sum((preds == 1) & (y_attack_test == 0))
        tp = np.sum((preds == 1) & (y_attack_test == 1))
        tn = np.sum((preds == 0) & (y_attack_test == 0))
        fn = np.sum((preds == 0) & (y_attack_test == 1))
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        if fpr <= fpr_threshold:
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            tpr_at_low_fpr = max(tpr_at_low_fpr, tpr)

    return {
        "auc": float(auc),
        "accuracy": float(acc),
        "tpr_low_fpr": float(tpr_at_low_fpr),
    }


def gradient_inversion_attack(model, grad_parameters, dummy_input_shape,
                               num_iters: int = 500, lr: float = 0.1):
    """梯度反演攻击 (Gradient Inversion Attack)

    从共享的模型梯度中尝试重建原始输入数据。
    通过优化一个随机初始化的输入，使其产生的梯度尽量匹配目标梯度。

    返回:
        - reconstructed: 重建的图像 (numpy array)
        - metrics: {"psnr": float, "ssim": float, "mse": float}
    """
    from skimage.metrics import structural_similarity as ssim

    device = next(model.parameters()).device
    model.eval()

    # 转换梯度格式为 tensor
    grad_tensors = []
    for g in grad_parameters:
        grad_tensors.append(torch.tensor(g, device=device))

    # 随机初始化一个 dummy 输入
    dummy = torch.randn(dummy_input_shape, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([dummy], lr=lr)

    for i in range(num_iters):
        optimizer.zero_grad()
        model.zero_grad()

        # 使用 dummy 输入做前向传播
        output = model(dummy)

        # 用随机标签计算"假"梯度
        fake_label = torch.zeros(dummy_input_shape[0], dtype=torch.long,
                                 device=device)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, fake_label)
        loss.backward()

        # 假梯度与目标梯度之间的 MSE 作为损失
        match_loss = 0.0
        for p, g_target in zip(model.parameters(), grad_tensors):
            if p.grad is not None:
                match_loss += nn.functional.mse_loss(p.grad, g_target)

        match_loss.backward()
        optimizer.step()

    # 转换为 numpy 并做反标准化可视化
    reconstructed = dummy.detach().cpu().numpy()
    # clamp 到合理范围
    reconstructed = np.clip(reconstructed, -3, 3)

    return reconstructed


def compute_inversion_metrics(original: np.ndarray,
                               reconstructed: np.ndarray) -> dict:
    """计算梯度反演重建图像的质量指标"""
    from skimage.metrics import structural_similarity as ssim

    # 归一化到 [0, 1]
    def normalize(x):
        x_min, x_max = x.min(), x.max()
        if x_max > x_min:
            return (x - x_min) / (x_max - x_min)
        return x - x_min

    orig_norm = normalize(original)
    recon_norm = normalize(reconstructed)

    mse = float(np.mean((orig_norm - recon_norm) ** 2))
    psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else 100.0

    # SSIM：逐样本计算后取平均
    ssim_vals = []
    for i in range(min(original.shape[0], 8)):  # 最多算 8 张
        if original.shape[1] == 1:  # 灰度图
            ssim_val = ssim(orig_norm[i, 0], recon_norm[i, 0],
                            data_range=1.0)
        else:  # RGB 图
            o = np.transpose(orig_norm[i], (1, 2, 0))
            r = np.transpose(recon_norm[i], (1, 2, 0))
            ssim_val = ssim(o, r, data_range=1.0, channel_axis=2)
        ssim_vals.append(ssim_val)

    avg_ssim = float(np.mean(ssim_vals)) if ssim_vals else 0.0

    return {"psnr": float(psnr), "ssim": avg_ssim, "mse": mse}
