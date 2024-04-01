import torch
import lightning as L
import timm
from torch.optim import Adam
from torchmetrics.classification import BinaryAccuracy

class ViTModel(L.LightningModule):
    def __init__(self, num_classes, model_name="vit_tiny_patch16_224_in21k", pretrained=True, lr=1e-4) -> None:
        super().__init__()
        self.pretrained_ViT = timm.create_model(model_name, pretrained=pretrained)
        self.lr = lr
        self.accuracy_metric = BinaryAccuracy()

        # Change the last layer to binary classification
        # Change the last layer to binary classification
        self.pretrained_ViT.head = torch.nn.Linear(
            in_features=self.pretrained_ViT.head.in_features, out_features=num_classes
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.pretrained_ViT(input_tensor)
    
    def training_step(self, batch, batch_idx):
        input_batch, target = batch
        # Calculate metrics
        logits = self(input_batch).squeeze()
        accuracy = self.accuracy_metric(logits, target)
        target = target.to(torch.float32)  # Convert labels to float32
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)

        self.log("train_loss", loss, on_epoch=True)
        self.log("train_accuracy", accuracy, prog_bar=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_batch, target = batch
        logits = self(input_batch).squeeze()
        accuracy = self.accuracy_metric(logits, target)
        target = target.to(torch.float32)  # Convert labels to float32
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
        self.log("validation_loss", loss, prog_bar=True, on_epoch=True)
        self.log("validation_accuracy", accuracy, prog_bar=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_batch, target = batch
        logits = self(input_batch).squeeze()
        accuracy = self.accuracy_metric(logits, target)
        target = target.to(torch.float32)  # Convert labels to float32
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
        self.log("test_loss", loss, prog_bar=True, on_epoch=True)
        self.log("test_accuracy", accuracy, prog_bar=True, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)

        return optimizer