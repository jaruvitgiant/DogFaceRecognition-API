import torch
import torch.optim as optim
from pytorch_metric_learning import losses, miners
import os
device = "cuda" if torch.cuda.is_available() else "cpu"

class FaceModelTrainer:
    def __init__(self, model, train_loader, device, num_classes, embedding_size=512):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device
        self.num_classes = num_classes
        self.embedding_size = embedding_size

    def train(self, epochs, save_path, progress_callback=None): # เพิ่ม callback
        self.model.train()
        
        optimizer = optim.Adam(self.model.parameters(), lr=5e-5)
        mining_func = miners.MultiSimilarityMiner(epsilon=0.1)
        loss_func = losses.SubCenterArcFaceLoss(
            num_classes=self.num_classes,
            embedding_size=self.embedding_size,
            margin=34.6, scale=64, sub_centers=4
        ).to(self.device)

        loss_optimizer = optim.Adam(loss_func.parameters(), lr=1e-5)

        for epoch in range(epochs):
            running_loss = 0.0
            for imgs, labels in self.train_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                loss_optimizer.zero_grad()
                
                embeddings = self.model(imgs)
                hard_pairs = mining_func(embeddings, labels)
                loss = loss_func(embeddings, labels, hard_pairs)
                
                loss.backward()
                optimizer.step()
                loss_optimizer.step()
                
                running_loss += loss.item()

            avg_loss = running_loss / len(self.train_loader)
            
            # --- ส่วนที่แก้ไข: ส่งข้อมูลจริงผ่าน Callback ---
            if progress_callback:
                # ส่ง Dict ข้อมูลที่ต้องการให้หน้าจอแสดงผล
                progress_callback({
                    "status": f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}",
                    "epoch": epoch + 1,
                    "loss": avg_loss
                })
            
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
            
            if (epoch + 1) % 20 == 0:
                torch.save(self.model.state_dict(), f"{save_path}_epoch_{epoch+1}.pt")
                
        return f"Training finished {epochs} epochs"