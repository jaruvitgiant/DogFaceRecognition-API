"""
services/visualization_service.py
------------------------------------
Business logic สำหรับ visualization (t-SNE, Confusion Matrix).

หลักการ SOLID ที่ใช้:
  - SRP: รับผิดชอบแค่การสร้าง visualization plots
  - OCP: เพิ่ม visualization type ใหม่ได้โดยเพิ่ม method ไม่ต้องแก้ที่อื่น
"""

import io
import base64
import numpy as np
import matplotlib
matplotlib.use("Agg")  # ใช้ Backend ที่ไม่มี GUI window
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


class VisualizationService:
    """
    สร้าง visualization plots และส่งกลับเป็น base64 encoded PNG string
    ทุก method เป็น stateless — รับข้อมูล คืน base64 string
    """
 
    # ───────────────────────────────────────────────
    # Public API
    # ───────────────────────────────────────────────

    def create_tsne_plot(self, X: np.ndarray, y: np.ndarray) -> str:
        """
        สร้าง t-SNE scatter plot จาก high-dimensional embeddings
        คืน base64 PNG string
        """
        if len(X) < 2:
            raise ValueError("ต้องมีข้อมูลอย่างน้อย 2 จุด")

        perplexity = min(20, len(X) - 1)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        X_transformed = tsne.fit_transform(X)

        return self._scatter_plot(
            X_transformed, y,
            title="t-SNE Visualization of Dog Embeddings",
            xlabel="t-SNE dimension 1",
            ylabel="t-SNE dimension 2",
        )

    def create_confusion_matrix(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[str, float]:
        """
        Train KNN classifier แล้วสร้าง confusion matrix
        คืน (base64 PNG string, accuracy float)
        """
        knn = KNeighborsClassifier(n_neighbors=min(2, len(X)))
        knn.fit(X, y)
        y_pred = knn.predict(X)

        acc = accuracy_score(y, y_pred)
        cm = confusion_matrix(y, y_pred)
        unique_labels = np.unique(y)

        try:
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=unique_labels,
                yticklabels=unique_labels,
            )
            plt.title(f"KNN Confusion Matrix (Accuracy: {acc:.2f})")
            plt.ylabel("Actual Dog ID")
            plt.xlabel("Predicted Dog ID")

            img_b64 = self._fig_to_base64()
        finally:
            self._close_plots()

        return img_b64, acc

    # ───────────────────────────────────────────────
    # Private helpers
    # ───────────────────────────────────────────────

    def _scatter_plot(
        self,
        X_2d: np.ndarray,
        y: np.ndarray,
        title: str,
        xlabel: str,
        ylabel: str,
    ) -> str:
        """สร้าง scatter plot และคืน base64 string"""
        try:
            plt.figure(figsize=(10, 7))
            scatter = plt.scatter(
                X_2d[:, 0], X_2d[:, 1],
                c=y, cmap="viridis", edgecolors="k", alpha=0.7,
            )
            plt.colorbar(scatter, label="Dog ID")
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)

            return self._fig_to_base64()
        finally:
            self._close_plots()

    def _fig_to_base64(self) -> str:
        """แปลง plt figure ปัจจุบันเป็น base64 PNG string"""
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    @staticmethod
    def _close_plots() -> None:
        """ปิด matplotlib figures เพื่อป้องกัน memory leak"""
        plt.clf()
        plt.close("all")
