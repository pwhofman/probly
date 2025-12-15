"""
Einfaches Beispiel zur Verwendung des First-Order Data Generators

Dieses Skript zeigt, wie man:
1. Einen einfachen Datensatz und ein Modell erstellt
2. First-Order Verteilungen generiert
3. Diese speichert und lädt
4. Mit FirstOrderDataset und DataLoader verwendet
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# Import der First-Order Generator Komponenten
from probly.data_generator.first_order_generator import (
    FirstOrderDataGenerator,
    FirstOrderDataset,
    output_fo_dataloader
)


# ============================================================================
# 1. DUMMY DATASET UND MODELL ERSTELLEN
# ============================================================================

class SimpleDataset(Dataset):
    """
    Ein einfacher Beispiel-Datensatz für Demonstrationszwecke.
    
    Generiert zufällige Eingabevektoren und gibt diese mit Labels zurück.
    """
    def __init__(self, n_samples=100, input_dim=10, n_classes=3):
        """
        Args:
            n_samples: Anzahl der Samples im Datensatz
            input_dim: Dimension der Eingabevektoren
            n_classes: Anzahl der Klassen (für Labels)
        """
        self.n_samples = n_samples
        self.input_dim = input_dim
        self.n_classes = n_classes
        
        # Generiere zufällige Daten
        torch.manual_seed(42)  # Für Reproduzierbarkeit
        self.data = torch.randn(n_samples, input_dim)
        self.labels = torch.randint(0, n_classes, (n_samples,))
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        # Gibt (input, label) zurück
        return self.data[idx], self.labels[idx]


class SimpleModel(nn.Module):
    """
    Ein einfaches neuronales Netzwerk für Klassifikation.
    """
    def __init__(self, input_dim=10, n_classes=3):
        """
        Args:
            input_dim: Dimension der Eingabe
            n_classes: Anzahl der Ausgabeklassen
        """
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes)
        )
    
    def forward(self, x):
        # Gibt Logits zurück (nicht Wahrscheinlichkeiten!)
        return self.network(x)


# ============================================================================
# 2. VERTEILUNGEN GENERIEREN
# ============================================================================

def generate_first_order_distributions():
    """
    Generiert First-Order Verteilungen aus einem Modell und Datensatz.
    """
    print("=" * 70)
    print("SCHRITT 1: First-Order Verteilungen generieren")
    print("=" * 70)
    
    # Parameter definieren
    n_samples = 100
    input_dim = 10
    n_classes = 3
    
    # Dataset erstellen
    print(f"\n Erstelle Datensatz mit {n_samples} Samples...")
    dataset = SimpleDataset(n_samples=n_samples, input_dim=input_dim, n_classes=n_classes)
    
    # Modell erstellen und initialisieren
    print(f" Erstelle und initialisiere Modell...")
    model = SimpleModel(input_dim=input_dim, n_classes=n_classes)
    model.eval()  # WICHTIG: Modell in Evaluationsmodus setzen!
    
    # Device auswählen
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f" Verwende Device: {device}")
    
    # Generator initialisieren
    print(f"\n⚙️  Initialisiere FirstOrderDataGenerator...")
    generator = FirstOrderDataGenerator(
        model=model,
        device=device,
        batch_size=32,
        output_mode='logits',  # Unser Modell gibt Logits aus
        model_name='simple_example_model'
    )
    
    # Verteilungen generieren
    print(f"\n Generiere Verteilungen (mit Fortschrittsanzeige)...")
    distributions = generator.generate_distributions(
        dataset,
        progress=True  # Zeigt Fortschritt an
    )
    
    print(f"\n Erfolgreich {len(distributions)} Verteilungen generiert!")
    
    # Beispiel-Verteilung anzeigen
    print(f"\n Beispiel-Verteilung für Sample 0:")
    print(f"   Wahrscheinlichkeiten: {[f'{p:.4f}' for p in distributions[0]]}")
    print(f"   Summe: {sum(distributions[0]):.6f} (sollte ≈ 1.0 sein)")
    
    return generator, distributions, dataset, n_classes


# ============================================================================
# 3. VERTEILUNGEN SPEICHERN UND LADEN
# ============================================================================

def save_and_load_distributions(generator, distributions, n_classes):
    """
    Demonstriert das Speichern und Laden von Verteilungen.
    """
    print("\n" + "=" * 70)
    print("SCHRITT 2: Verteilungen speichern und laden")
    print("=" * 70)
    
    # Pfad für Output-Datei
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    save_path = output_dir / "example_first_order_dists.json"
    
    # Metadaten definieren
    metadata = {
        'dataset': 'SimpleDataset',
        'n_samples': len(distributions),
        'n_classes': n_classes,
        'note': 'Generated for documentation example',
        'purpose': 'demonstration'
    }
    
    # Speichern
    print(f"\n Speichere Verteilungen nach: {save_path}")
    generator.save_distributions(
        path=save_path,
        distributions=distributions,
        meta=metadata
    )
    print(f" Erfolgreich gespeichert!")
    
    # Laden
    print(f"\n📂 Lade Verteilungen von: {save_path}")
    loaded_distributions, loaded_metadata = generator.load_distributions(save_path)
    
    print(f" Erfolgreich geladen!")
    print(f"\n Metadaten:")
    for key, value in loaded_metadata.items():
        print(f"   - {key}: {value}")
    
    # Verifizierung
    print(f"\n Verifizierung:")
    print(f"   - Anzahl Verteilungen: {len(loaded_distributions)}")
    print(f"   - Originale == Geladene: {distributions == loaded_distributions}")
    
    return loaded_distributions, loaded_metadata


# ============================================================================
# 4. FIRSTORDERDATASET VERWENDEN
# ============================================================================

def use_first_order_dataset(dataset, distributions):
    """
    Zeigt die Verwendung von FirstOrderDataset.
    """
    print("\n" + "=" * 70)
    print("SCHRITT 3: FirstOrderDataset verwenden")
    print("=" * 70)
    
    # FirstOrderDataset erstellen
    print(f"\n Erstelle FirstOrderDataset...")
    fo_dataset = FirstOrderDataset(
        base_dataset=dataset,
        distributions=distributions
    )
    
    print(f" Dataset erstellt mit {len(fo_dataset)} Samples")
    
    # Ein Sample abrufen
    print(f"\n Sample 0 abrufen:")
    sample = fo_dataset[0]
    
    # Sample kann (input, label, distribution) oder (input, distribution) sein
    if len(sample) == 3:
        input_tensor, label, distribution = sample
        print(f"   - Input shape: {input_tensor.shape}")
        print(f"   - Label: {label}")
        print(f"   - Distribution shape: {distribution.shape}")
        print(f"   - Distribution: {[f'{p:.4f}' for p in distribution.tolist()]}")
    else:
        input_tensor, distribution = sample
        print(f"   - Input shape: {input_tensor.shape}")
        print(f"   - Distribution shape: {distribution.shape}")
        print(f"   - Distribution: {[f'{p:.4f}' for p in distribution.tolist()]}")
    
    return fo_dataset


# ============================================================================
# 5. DATALOADER MIT FIRST-ORDER VERTEILUNGEN
# ============================================================================

def use_dataloader_with_distributions(dataset, distributions):
    """
    Demonstriert die Verwendung eines DataLoaders mit First-Order Verteilungen.
    """
    print("\n" + "=" * 70)
    print("SCHRITT 4: DataLoader mit First-Order Verteilungen")
    print("=" * 70)
    
    # DataLoader erstellen
    print(f"\n Erstelle DataLoader mit batch_size=16...")
    fo_loader = output_fo_dataloader(
        base_dataset=dataset,
        distributions=distributions,
        batch_size=16,
        shuffle=True,
        num_workers=0,  # Für Windows Kompatibilität
        pin_memory=False
    )
    
    print(f" DataLoader erstellt!")
    
    # Ersten Batch abrufen
    print(f"\n Rufe ersten Batch ab...")
    batch = next(iter(fo_loader))
    
    if len(batch) == 3:
        inputs, labels, distributions_batch = batch
        print(f"   - Inputs shape: {inputs.shape}")
        print(f"   - Labels shape: {labels.shape}")
        print(f"   - Distributions shape: {distributions_batch.shape}")
    else:
        inputs, distributions_batch = batch
        print(f"   - Inputs shape: {inputs.shape}")
        print(f"   - Distributions shape: {distributions_batch.shape}")
    
    # Zeige wie man über den Loader iteriert
    print(f"\n Iteriere über alle Batches...")
    for batch_idx, batch in enumerate(fo_loader):
        if batch_idx == 0:
            print(f"   Batch {batch_idx}: {len(batch)} Tensoren")
    
    total_batches = len(fo_loader)
    print(f"   ... (insgesamt {total_batches} Batches)")
    
    return fo_loader


# ============================================================================
# 6. TRAINING MIT SOFT TARGETS (BONUS)
# ============================================================================

def train_with_soft_targets(fo_loader, input_dim, n_classes, epochs=3):
    """
    Zeigt ein einfaches Training mit First-Order Verteilungen als Soft Targets.
    """
    print("\n" + "=" * 70)
    print("SCHRITT 5: Training mit Soft Targets (Bonus)")
    print("=" * 70)
    
    # Student-Modell erstellen
    print(f"\n🎓 Erstelle Student-Modell...")
    student_model = SimpleModel(input_dim=input_dim, n_classes=n_classes)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    student_model = student_model.to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)
    
    print(f"⚙️  Training für {epochs} Epochen...")
    
    for epoch in range(epochs):
        student_model.train()
        epoch_loss = 0.0
        n_batches = 0
        
        for batch in fo_loader:
            # Batch entpacken
            if len(batch) == 3:
                inputs, labels, target_distributions = batch
            else:
                inputs, target_distributions = batch
            
            # Zu Device verschieben
            inputs = inputs.to(device)
            target_distributions = target_distributions.to(device)
            
            # Forward pass
            logits = student_model(inputs)
            
            # KL Divergenz Loss zwischen Modell-Ausgabe und Ziel-Verteilungen
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            loss = torch.nn.functional.kl_div(
                log_probs, 
                target_distributions, 
                reduction='batchmean'
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        print(f"   Epoch {epoch + 1}/{epochs} - Durchschnittlicher Loss: {avg_loss:.4f}")
    
    print(f"\n Training abgeschlossen!")
    
    return student_model


# ============================================================================
# MAIN FUNKTION
# ============================================================================

def main():
    """
    Hauptfunktion die alle Schritte ausführt.
    """
    print("\n" + "=" * 70)
    print(" First-Order Data Generator - Einfaches Beispiel")
    print("=" * 70)
    
    # Schritt 1: Verteilungen generieren
    generator, distributions, dataset, n_classes = generate_first_order_distributions()
    
    # Schritt 2: Speichern und Laden
    loaded_distributions, metadata = save_and_load_distributions(
        generator, distributions, n_classes
    )
    
    # Schritt 3: FirstOrderDataset verwenden
    fo_dataset = use_first_order_dataset(dataset, loaded_distributions)
    
    # Schritt 4: DataLoader verwenden
    fo_loader = use_dataloader_with_distributions(dataset, loaded_distributions)
    
    # Schritt 5: Training (optional)
    print("\n" + "=" * 70)
    print("🎓 Möchten Sie ein kurzes Training demonstrieren? (Optional)")
    print("=" * 70)
    print("Hinweis: Dies zeigt, wie man mit den generierten Verteilungen")
    print("         als 'Soft Targets' trainieren kann.")
    
    # Für das Beispiel trainieren wir einfach
    student_model = train_with_soft_targets(
        fo_loader, 
        input_dim=10, 
        n_classes=n_classes, 
        epochs=2
    )
    
    print("\n" + "=" * 70)
    print(" Beispiel erfolgreich abgeschlossen!")
    print("=" * 70)
    print("\nZusammenfassung:")
    print(f"  ✓ {len(distributions)} Verteilungen generiert")
    print(f"  ✓ Verteilungen gespeichert und geladen")
    print(f"  ✓ FirstOrderDataset erstellt")
    print(f"  ✓ DataLoader verwendet")
    print(f"  ✓ Student-Modell trainiert")
    print("\n Weitere Informationen: Siehe docs/data_generation_guide.md")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()