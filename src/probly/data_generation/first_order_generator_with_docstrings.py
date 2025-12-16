"""First-Order Data Generator for Uncertainty Quantification.

Dieses Modul stellt Werkzeuge zur Generierung approximativer bedingter
Wahrscheinlichkeitsverteilungen p(Y|X) aus vortrainierten Modellen bereit.

This module provides tools for generating approximate conditional probability
distributions p(Y|X) from pre-trained models.

Classes:
    FirstOrderDataGenerator: Hauptklasse zur Generierung von First-Order Verteilungen
    FirstOrderDataset: PyTorch Dataset-Wrapper für First-Order Verteilungen

Functions:
    output_fo_dataloader: Erstellt DataLoader mit First-Order Verteilungen
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any, cast
import warnings

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


def _is_probabilities(x: torch.Tensor, atol: float = 1e-4) -> bool:
    """
    Prüft ob ein Tensor Wahrscheinlichkeiten entlang der letzten Dimension darstellt.
    
    Check if a tensor represents probabilities along the last dimension.
    
    Bedingungen / Conditions:
    - Alle Werte in [0, 1] / All values in [0, 1]
    - Zeilen summieren sich zu ~1 (innerhalb atol) / Rows sum to ~1 (within atol)
    
    Args:
        x: Input tensor zu prüfen / Input tensor to check
        atol: Absolute Toleranz für Summierung / Absolute tolerance for summation
    
    Returns:
        bool: True falls Tensor Wahrscheinlichkeiten repräsentiert / True if tensor represents probabilities
    
    Examples:
        >>> probs = torch.tensor([[0.2, 0.3, 0.5], [0.1, 0.6, 0.3]])
        >>> _is_probabilities(probs)
        True
        >>> logits = torch.tensor([[2.0, 1.0, 3.0]])
        >>> _is_probabilities(logits)
        False
    """
    if x.numel() == 0:
        return False
    min_ok = bool(torch.all(x >= -atol))
    max_ok = bool(torch.all(x <= 1 + atol))
    if not (min_ok and max_ok):
        return False
    sums = x.sum(dim=-1)
    return bool(torch.allclose(sums, torch.ones_like(sums), atol=atol, rtol=0))


@dataclass
class FirstOrderDataGenerator:
    """
    Generator für First-Order Wahrscheinlichkeitsverteilungen.
    
    Generator for first-order probability distributions.
    
    Diese Klasse nimmt ein vortrainiertes Modell und einen Datensatz und generiert
    für jedes Sample eine Wahrscheinlichkeitsverteilung, die als Approximation der
    wahren bedingten Verteilung p(Y|X) dient.
    
    This class takes a pre-trained model and dataset and generates a probability
    distribution for each sample, serving as an approximation of the true
    conditional distribution p(Y|X).
    
    Attributes:
        model: Aufrufbares Objekt (meist torch.nn.Module), das Inputs auf Logits/Probs abbildet
               Callable object (typically torch.nn.Module) that maps inputs to logits/probs
        device: Gerät für Inferenz ('cpu' oder 'cuda') / Device for inference
        batch_size: Batch-Größe für Verarbeitung / Batch size for processing
        output_mode: Output-Modus - 'auto', 'logits' oder 'probs'
        output_transform: Optional benutzerdefinierte Transformationsfunktion
                         Optional custom transformation function
        input_getter: Optional Funktion zum Extrahieren der Modelleingabe
                     Optional function for extracting model input
        model_name: Optionaler Modell-Identifier / Optional model identifier
    
    Examples:
        >>> # Einfache Verwendung / Simple usage
        >>> model = torch.load('pretrained_model.pt')
        >>> generator = FirstOrderDataGenerator(model=model, device='cuda')
        >>> distributions = generator.generate_distributions(dataset)
        >>> generator.save_distributions('output.json', distributions)
        
        >>> # Mit benutzerdefinierten Optionen / With custom options
        >>> generator = FirstOrderDataGenerator(
        ...     model=model,
        ...     device='cpu',
        ...     batch_size=32,
        ...     output_mode='logits',
        ...     model_name='resnet50_v1'
        ... )
    """

    model: Callable[..., Any]
    device: str = "cpu"
    batch_size: int = 64
    output_mode: str = "auto"
    output_transform: Callable[[torch.Tensor], torch.Tensor] | None = None
    input_getter: Callable[[Any], Any] | None = None
    model_name: str | None = None

    def to_device(self, x: object) -> object:
        """
        Verschiebt Tensor(en) auf das konfigurierte Gerät.
        
        Move tensor(s) to the configured device.
        
        Unterstützt verschachtelte Strukturen (Listen, Tupel, Dictionaries).
        Supports nested structures (lists, tuples, dictionaries).
        
        Args:
            x: Tensor oder verschachtelte Struktur von Tensoren
               Tensor or nested structure of tensors
        
        Returns:
            object: Eingabe mit Tensoren auf Ziel-Gerät
                   Input with tensors on target device
        
        Examples:
            >>> generator = FirstOrderDataGenerator(model=model, device='cuda')
            >>> x = torch.randn(10, 3)
            >>> x_cuda = generator.to_device(x)  # x ist jetzt auf CUDA / x is now on CUDA
        """
        if isinstance(x, torch.Tensor):
            return x.to(self.device)
        if isinstance(x, (list, tuple)):
            return type(x)(self.to_device(xx) for xx in x)
        if isinstance(x, Mapping):
            return {k: self.to_device(v) for k, v in x.items()}
        return x

    def to_probs(self, outputs: torch.Tensor) -> torch.Tensor:
        """
        Konvertiert Modellausgaben zu Wahrscheinlichkeiten.
        
        Convert model outputs to probabilities.
        
        Verwendet entweder output_transform, output_mode oder Auto-Detection.
        Uses either output_transform, output_mode, or auto-detection.
        
        Args:
            outputs: Modellausgaben (Logits oder Probs) / Model outputs (logits or probs)
        
        Returns:
            torch.Tensor: Wahrscheinlichkeitsverteilungen (summieren zu 1)
                         Probability distributions (sum to 1)
        
        Raises:
            ValueError: Falls outputs ungültiges Format hat / If outputs have invalid format
        
        Examples:
            >>> logits = torch.tensor([[2.0, 1.0, 3.0]])
            >>> probs = generator.to_probs(logits)
            >>> print(probs.sum(dim=-1))  # tensor([1.])
        """
        if self.output_transform is not None:
            return self.output_transform(outputs)

        mode = self.output_mode.lower()
        if mode == "probs":
            return outputs
        if mode == "logits":
            return F.softmax(outputs, dim=-1)
        # auto
        return outputs if _is_probabilities(outputs) else F.softmax(outputs, dim=-1)

    def prepares_batch_inp(self, sample: object) -> object:
        """
        Bereitet Modelleingabe aus Dataset-Sample vor.
        
        Prepare model input from dataset sample.
        
        Args:
            sample: Sample aus dem Dataset / Sample from dataset
        
        Returns:
            object: Vorbereitete Modelleingabe / Prepared model input
        
        Examples:
            >>> # Dataset gibt (input, label) zurück / Dataset returns (input, label)
            >>> sample = (torch.randn(3, 32, 32), 5)
            >>> inp = generator.prepares_batch_inp(sample)  # torch.randn(3, 32, 32)
        """
        if self.input_getter is not None:
            return self.input_getter(sample)
        if isinstance(sample, (list, tuple)) and len(sample) >= 1:
            return sample[0]
        return sample

    def extract_input(self, sample: object) -> object:
        """
        Extrahiert Modelleingabe aus Dataset-Sample.
        
        Extract model input from dataset sample.
        
        Standard-Konventionen: (input, target) oder nur input
        Default conventions: (input, target) or input only
        
        Args:
            sample: Sample aus dem Dataset / Sample from dataset
        
        Returns:
            object: Extrahierte Modelleingabe / Extracted model input
        """
        if self.input_getter is not None:
            return self.input_getter(sample)
        if isinstance(sample, (list, tuple)) and len(sample) >= 1:
            return sample[0]
        return sample

    @torch.no_grad()
    def generate_distributions(
        self,
        dataset_or_loader: object,
        *,
        progress: bool = True,
    ) -> dict[int, list[float]]:
        """
        Generiert Wahrscheinlichkeitsverteilungen für alle Samples.
        
        Generate probability distributions for all samples.
        
        Dies ist die Hauptmethode zur Generierung von First-Order Daten.
        This is the main method for generating first-order data.
        
        Args:
            dataset_or_loader: torch.utils.data.Dataset oder DataLoader
                              torch.utils.data.Dataset or DataLoader
            progress: Ob Fortschritt angezeigt werden soll
                     Whether to display progress
        
        Returns:
            dict[int, list[float]]: Mapping von Dataset-Index zu Wahrscheinlichkeitsliste
                                   Mapping from dataset index to probability list
        
        Raises:
            TypeError: Falls Modell keinen torch.Tensor zurückgibt
                      If model does not return a torch.Tensor
        
        Examples:
            >>> generator = FirstOrderDataGenerator(model=model, device='cpu')
            >>> distributions = generator.generate_distributions(dataset, progress=True)
            >>> # distributions = {0: [0.1, 0.3, 0.6], 1: [0.2, 0.5, 0.3], ...}
            >>> print(f"Generiert {len(distributions)} Verteilungen")
        
        Notes:
            - Modell wird automatisch in eval() Modus gesetzt / Model automatically set to eval()
            - Gradient-Berechnung ist deaktiviert / Gradient computation is disabled
            - Bei DataLoadern wird shuffle=False empfohlen / shuffle=False recommended for DataLoaders
        """
        # Loader vorbereiten / Prepare loader
        if isinstance(dataset_or_loader, torch.utils.data.DataLoader):
            loader = dataset_or_loader
            dataset_len = len(loader.dataset) if loader.dataset is not None else None
        else:
            dataset = cast("Dataset", dataset_or_loader)
            dataset_len = len(dataset)
            loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        # Modell vorbereiten / Prepare model
        self.model = self.model.to(self.device) if hasattr(self.model, "to") else self.model
        if hasattr(self.model, "eval"):
            self.model.eval()

        distributions: dict[int, list[float]] = {}
        start_idx = 0
        total_batches = len(loader)
        
        for batch_idx, batch in enumerate(loader):
            inpt = self.prepares_batch_inp(batch)
            inpt = self.to_device(inpt)
            outputs = self.model(inpt)
            
            if not isinstance(outputs, torch.Tensor):
                msg = "Model must return a torch.Tensor (logits or probs)."
                raise TypeError(msg)
            
            probs = self.to_probs(outputs)
            if probs.ndim == 1:
                probs = probs.unsqueeze(0)

            probs_np = probs.detach().cpu().numpy()

            batch_size = probs_np.shape[0]
            for i in range(batch_size):
                idx = start_idx + i
                distributions[idx] = probs_np[i].tolist()

            start_idx += batch_size
            if progress:
                logger.info("[FirstOrderDataGenerator] Batch %d/%d", batch_idx + 1, total_batches)

        if progress:
            logger.info("[FirstOrderDataGenerator] Finished %d batches", total_batches)

        # Warnung bei Längen-Mismatch / Warning for length mismatch
        if dataset_len is not None and len(distributions) != dataset_len:
            warnings.warn(
                (
                    f"[FirstOrderDataGenerator] generated {len(distributions)} distributions, "
                    f"but dataset length is {dataset_len}."
                ),
                stacklevel=2,
            )

        return distributions

    def save_distributions(
        self,
        path: str | Path,
        distributions: Mapping[int, Iterable[float]],
        *,
        meta: dict[str, Any] | None = None,
    ) -> None:
        """
        Speichert Verteilungen und Metadaten als JSON-Datei.
        
        Save distributions and metadata as JSON file.
        
        Args:
            path: Zielpfad für JSON-Datei / Target path for JSON file
            distributions: Zu speichernde Verteilungen / Distributions to save
            meta: Optionale Metadaten / Optional metadata
        
        Raises:
            IOError: Bei Schreibfehlern / On write errors
        
        Examples:
            >>> generator.save_distributions(
            ...     'output/dists.json',
            ...     distributions,
            ...     meta={'dataset': 'MNIST', 'accuracy': 0.95}
            ... )
        
        Notes:
            - Erstellt Verzeichnis falls nicht vorhanden / Creates directory if not exists
            - Verwendet UTF-8 Encoding / Uses UTF-8 encoding
            - Speichert Indices als Strings (JSON-kompatibel) / Saves indices as strings
        """
        path = Path(path)
        serializable = {
            "meta": {
                "model_name": self.model_name,
                **(meta or {}),
            },
            "distributions": {str(k): list(v) for k, v in distributions.items()},
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False)

    def load_distributions(self, path: str | Path) -> tuple[dict[int, list[float]], dict[str, Any]]:
        """
        Lädt Verteilungen und Metadaten aus JSON-Datei.
        
        Load distributions and metadata from JSON file.
        
        Args:
            path: Quellpfad der JSON-Datei / Source path of JSON file
        
        Returns:
            tuple: (distributions, metadata)
                distributions: dict[int, list[float]] - Verteilungen
                metadata: dict[str, Any] - Gespeicherte Metadaten / Saved metadata
        
        Raises:
            FileNotFoundError: Falls Datei nicht existiert / If file does not exist
            json.JSONDecodeError: Bei ungültigem JSON / On invalid JSON
        
        Examples:
            >>> dists, meta = generator.load_distributions('output/dists.json')
            >>> print(f"Modell: {meta['model_name']}")
            >>> print(f"Samples: {len(dists)}")
        
        Notes:
            - String-Indices werden zu Integers konvertiert / String indices converted to integers
            - Kompatibel mit save_distributions() Format / Compatible with save_distributions() format
        """
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        meta = obj.get("meta", {}) or {}
        dists_raw = obj.get("distributions", {}) or {}
        # Konvertiere Schlüssel zurück zu int / Convert keys back to int
        distributions: dict[int, list[float]] = {int(k): list(v) for k, v in dists_raw.items()}
        return distributions, meta


class FirstOrderDataset(Dataset):
    """
    PyTorch Dataset-Wrapper für First-Order Verteilungen.
    
    PyTorch Dataset wrapper for first-order distributions.
    
    Kombiniert einen existierenden Datensatz mit First-Order Verteilungen.
    Combines an existing dataset with first-order distributions.
    
    Gibt zurück / Returns:
    - (input, distribution) falls base_dataset nur input zurückgibt
      (input, distribution) if base_dataset returns only input
    - (input, label, distribution) falls base_dataset (input, label) zurückgibt
      (input, label, distribution) if base_dataset returns (input, label)
    
    Attributes:
        base_dataset: Ursprünglicher PyTorch Dataset / Original PyTorch dataset
        distributions: Mapping von Index zu Verteilung / Mapping from index to distribution
        input_getter: Optional Funktion für Input-Extraktion / Optional function for input extraction
    
    Examples:
        >>> # Mit Labels / With labels
        >>> fo_dataset = FirstOrderDataset(base_dataset, distributions)
        >>> input, label, dist = fo_dataset[0]
        
        >>> # Ohne Labels / Without labels
        >>> fo_dataset = FirstOrderDataset(input_only_dataset, distributions)
        >>> input, dist = fo_dataset[0]
    
    Warnings:
        - Länge von distributions sollte mit base_dataset übereinstimmen
          Length of distributions should match base_dataset
        - Indices müssen konsistent sein / Indices must be consistent
    """

    def __init__(
        self,
        base_dataset: Dataset,
        distributions: Mapping[int, Iterable[float]],
        input_getter: Callable[[object], object] | None = None,
    ) -> None:
        """
        Initialisiert FirstOrderDataset mit base_dataset und Verteilungen.
        
        Initialize FirstOrderDataset with base_dataset and distributions.
        
        Args:
            base_dataset: Ursprünglicher PyTorch Dataset / Original PyTorch dataset
            distributions: Index-aligned Verteilungen / Index-aligned distributions
            input_getter: Optional Funktion zum Extrahieren der Eingabe
                         Optional function for extracting input
        
        Warns:
            Wenn Längen nicht übereinstimmen / When lengths do not match
        """
        self.base_dataset = base_dataset
        self.distributions: dict[int, list[float]] = {int(k): list(v) for k, v in distributions.items()}
        self.input_getter = input_getter

        # Soft check nur falls base_dataset Länge unterstützt
        # Soft check only if base_dataset supports length
        try:
            n = len(base_dataset)
            if len(self.distributions) != n:
                warnings.warn(
                    (
                        f"[FirstOrderDataset] distributions count {len(self.distributions)} "
                        f"does not match dataset length {n}."
                    ),
                    stacklevel=2,
                )
        except TypeError:
            warnings.warn(
                "[FirstOrderDataset] base_dataset has no length; validation being skipped.",
                stacklevel=2,
            )

    def __len__(self) -> int:
        """
        Gibt Anzahl der Samples zurück.
        
        Return number of samples.
        
        Returns:
            int: Anzahl der Samples / Number of samples
        """
        return len(self.base_dataset)

    def _get_input(self, sample: object) -> object:
        """
        Extrahiert Input aus Sample mit input_getter falls vorhanden.
        
        Extract input from sample using input_getter if provided.
        
        Args:
            sample: Sample aus base_dataset / Sample from base_dataset
        
        Returns:
            object: Extrahierte Eingabe / Extracted input
        """
        if self.input_getter is not None:
            return self.input_getter(sample)
        if isinstance(sample, (list, tuple)) and len(sample) >= 1:
            return sample[0]
        return sample

    def __getitem__(self, idx: int) -> object:
        """
        Gibt Input (+ optional Label) und Verteilung bei Index zurück.
        
        Return input (+ optional label) and distribution at index.
        
        Args:
            idx: Index des Samples / Index of sample
        
        Returns:
            tuple: (input, distribution) oder (input, label, distribution)
                  (input, distribution) or (input, label, distribution)
        
        Raises:
            KeyError: Falls keine Verteilung für Index existiert
                     If no distribution exists for index
        
        Examples:
            >>> sample = fo_dataset[5]
            >>> if len(sample) == 3:
            ...     input, label, dist = sample
            >>> else:
            ...     input, dist = sample
        """
        sample = self.base_dataset[idx]
        dist = self.distributions.get(idx)
        if dist is None:
            msg = f"No distribution for index {idx}."
            raise KeyError(msg)

        dist_tensor = torch.tensor(dist, dtype=torch.float32)
        if isinstance(sample, (list, tuple)) and len(sample) >= 2:
            inp, lbl = sample[0], sample[1]
            return inp, lbl, dist_tensor
        inp = self._get_input(sample)
        return inp, dist_tensor


def output_fo_dataloader(
    base_dataset: Dataset,
    distributions: Mapping[int, Iterable[float]],
    *,
    batch_size: int = 64,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    input_getter: Callable[[Any], Any] | None = None,
) -> DataLoader:
    """
    Erstellt DataLoader der Inputs (+ Labels) mit First-Order Verteilungen paart.
    
    Create DataLoader that pairs inputs (+ labels) with first-order distributions.
    
    Dies ist eine Convenience-Funktion, die FirstOrderDataset und DataLoader kombiniert.
    This is a convenience function that combines FirstOrderDataset and DataLoader.
    
    Args:
        base_dataset: Ursprünglicher PyTorch Dataset / Original PyTorch dataset
        distributions: Index-aligned Verteilungen / Index-aligned distributions
        batch_size: Batch-Größe / Batch size (default: 64)
        shuffle: Ob Daten gemischt werden sollen / Whether to shuffle data (default: False)
        num_workers: Anzahl Worker-Prozesse / Number of worker processes (default: 0)
        pin_memory: Ob Memory Pinning verwendet werden soll / Whether to use memory pinning (default: False)
        input_getter: Optional Funktion für Input-Extraktion / Optional function for input extraction
    
    Returns:
        DataLoader: PyTorch DataLoader mit First-Order Verteilungen
                   PyTorch DataLoader with first-order distributions
    
    Examples:
        >>> # Einfache Verwendung / Simple usage
        >>> fo_loader = output_fo_dataloader(dataset, distributions, batch_size=32)
        >>> for batch in fo_loader:
        ...     inputs, labels, dists = batch
        ...     # Training hier / Training here
        
        >>> # Mit erweiterten Optionen / With advanced options
        >>> fo_loader = output_fo_dataloader(
        ...     base_dataset=dataset,
        ...     distributions=distributions,
        ...     batch_size=32,
        ...     shuffle=True,
        ...     num_workers=4,
        ...     pin_memory=True
        ... )
    
    Notes:
        - Verwendet FirstOrderDataset intern / Uses FirstOrderDataset internally
        - Kompatibel mit allen PyTorch DataLoader Features / Compatible with all PyTorch DataLoader features
        - Bei Windows: num_workers=0 empfohlen / On Windows: num_workers=0 recommended
    """
    firstorderdataset = FirstOrderDataset(base_dataset, distributions, input_getter=input_getter)
    return DataLoader(
        firstorderdataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
