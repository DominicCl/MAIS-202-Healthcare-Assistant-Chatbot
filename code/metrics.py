import torch

def multilabel_metrics_from_logits(logits, labels, threshold=0.5):
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).float()

        top1_pred = torch.argmax(logits, dim=1)
        top1_true = torch.argmax(labels, dim=1)
        acc = (top1_pred == top1_true).float().mean().item()

        eps = 1e-8
        tp = (preds * labels).sum(dim=0)
        fp = ((preds == 1) & (labels == 0)).sum(dim=0)
        fn = ((preds == 0) & (labels == 1)).sum(dim=0)

        precision_c = tp / (tp + fp + eps)
        recall_c = tp / (tp + fn + eps)
        f1_c = 2 * precision_c * recall_c / (precision_c + recall_c + eps)

        return acc, precision_c.mean().item(), recall_c.mean().item(), f1_c.mean().item()

class MetricsEvaluator:
    def __init__(self, device):
        self.device = device

    def evaluate(self, model, data_loader, threshold=0.5, debug=False):
        model.eval()
        all_logits, all_labels = [], []
        with torch.no_grad():
            for batch in data_loader:
                x = batch["features"].to(self.device)
                y = batch["labels"].to(self.device)
                logits = model(x)
                all_logits.append(logits.cpu())
                all_labels.append(y.cpu())

        logits = torch.cat(all_logits)
        labels = torch.cat(all_labels)

        if debug:
            probs = torch.sigmoid(logits)
            print(f"[DEBUG] Shapes: logits={logits.shape}, labels={labels.shape}")
            print(f"Example probs[0][:5]: {probs[0][:5]}")

        return multilabel_metrics_from_logits(logits, labels, threshold)
