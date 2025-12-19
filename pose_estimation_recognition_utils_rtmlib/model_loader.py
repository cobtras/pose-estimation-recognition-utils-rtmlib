# Copyright 2025 Jonas David Stephan, Nathalie Dollmann
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
model_loader.py

This module provides a class to intelligently load and cache models from the Hugging Face Hub.

Author: Jonas David Stephan, Nathalie Dollmann
Date: 2025-12-18
License: Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0)
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional
from huggingface_hub import hf_hub_download, HfApi, model_info
from huggingface_hub.constants import HF_HUB_CACHE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelLoader:
    """
    Eine Klasse zum intelligenten Laden und Cachen von Modellen vom Hugging Face Hub.
    """

    def __init__(
        self,
        repo_id: str,
        model_filename: str,
        cache_dir: Optional[os.PathLike] = None,
        local_model_dir: Optional[os.PathLike] = None
    ):
        """
        Initialisiert den ModelLoader.

        Args:
            repo_id (str): Die HF Repo-ID (z.B. 'fhswf/rtm133lifting').
            model_filename (str): Der Dateiname des Modells im Repo (z.B. 'rtm133lifting.pth').
            cache_dir (os.PathLike, optional): Basis-Cache-Ordner. Standardmäßig wird der Standard-HF-Cache oder
                                                ~/.cache/huggingface/hub verwendet.
            local_model_dir (os.PathLike, optional): Spezifischer Ordner für lokales Modell. Überschreibt die Cache-Struktur.
        """
        self.repo_id = repo_id
        self.model_filename = model_filename
        self.hf_api = HfApi()

        if local_model_dir is not None:
            self.model_dir = Path(local_model_dir)
        else:
            if cache_dir is not None:
                base_cache = Path(cache_dir)
            else:
                base_cache = Path(HF_HUB_CACHE)

            safe_repo_name = repo_id.replace("/", "--")
            self.model_dir = base_cache / f"models--{safe_repo_name}"

            self.model_dir.mkdir(parents=True, exist_ok=True)

            self.local_model_path = self.model_dir / model_filename
            self.metadata_path = self.model_dir / "model_metadata.json"

            self.local_metadata = self._load_local_metadata()

    def _load_local_metadata(self) -> dict:
        """
        Lädt die gespeicherten Metadaten des lokal gecachten Modells.

        Returns
            dict: Die Metadaten als Dictionary.
        """
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Metadaten-Datei {self.metadata_path} ist beschädigt.")
        return {}

    def _save_local_metadata(self, metadata: dict):
        """
        Speichert Metadaten des lokal gecachten Modells.
        
        Args:
            metadata (dict): Die zu speichernden Metadaten.
        """
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def _get_remote_metadata(self) -> Optional[dict]:
        """
        Ruft die aktuellen Metadaten des Modells vom Hugging Face Hub ab.
        
        Returns:
            dict: Die Remote-Metadaten als Dictionary, oder None bei Fehlern.
        """
        try:
            info = model_info(self.repo_id, files_metadata=True)

            for file_info in info.siblings:
                if file_info.rfilename == self.model_filename:
                    latest_commit_hash = info.sha 

                    return {
                        "repo_id": self.repo_id,
                        "model_filename": self.model_filename,
                        "last_remote_commit": latest_commit_hash,
                    }
            logger.error(f"Datei '{self.model_filename}' nicht im Repository '{self.repo_id}' gefunden.")
            return None
        except Exception as e:
            logger.error(f"Fehler beim Abrufen der Remote-Metadaten: {e}")
            return None

    def check_for_update(self) -> bool:
        """
        Prüft, ob eine neuere Version des Modells auf dem Hub verfügbar ist.

        Returns:
            bool: True, wenn ein Update verfügbar ist oder kein lokales Modell existiert.
                  False, wenn die lokale Version aktuell ist.
        """
        if not self.local_model_path.exists():
            logger.info("Keine lokale Modell-Datei gefunden. Download erforderlich.")
            return True

        remote_meta = self._get_remote_metadata()
        if remote_meta is None:
            logger.warning("Konnte Remote-Status nicht prüfen. Verwende lokale Datei.")
            return False

        local_commit = self.local_metadata.get("last_remote_commit")
        remote_commit = remote_meta.get("last_remote_commit")

        if local_commit != remote_commit:
            logger.info(f"Update verfügbar! Lokaler Commit: {local_commit}, Remote Commit: {remote_commit}")
            return True
        else:
            logger.info("Lokales Modell ist auf dem neuesten Stand.")
            return False

    def download_model(self, force_download: bool = False) -> Path:
        """
        Lädt das Modell herunter, wenn nötig.

        Args:
            force_download (bool): Wenn True, wird das Modell auch bei vorhandener,
                                   aktueller lokaler Datei neu heruntergeladen.

        Returns:
            Path: Der Pfad zur lokalen Modell-Datei.
        """
        needs_download = force_download or self.check_for_update()

        if needs_download:
            logger.info(f"Lade Modell herunter in: {self.local_model_path}")
            try:
                downloaded_path = hf_hub_download(
                    repo_id=self.repo_id,
                    filename=self.model_filename,
                    cache_dir=self.model_dir.parent, 
                    force_download=force_download
                )

                import shutil
                shutil.copy2(downloaded_path, self.local_model_path)
                logger.info(f"Modell erfolgreich nach {self.local_model_path} heruntergeladen.")

                remote_meta = self._get_remote_metadata()
                if remote_meta:
                    self.local_metadata = remote_meta
                    self._save_local_metadata(self.local_metadata)
                else:
                    logger.warning("Modell heruntergeladen, aber Metadaten konnten nicht gespeichert werden.")

            except Exception as e:
                logger.error(f"Download fehlgeschlagen: {e}")
                raise
        else:
            logger.info(f"Verwende vorhandenes Modell unter: {self.local_model_path}")

        return self.local_model_path

    def load_model(self, force_download: bool = False, **model_load_kwargs):
        """
        Hauptmethode: Stellt sicher, dass das aktuellste Modell vorhanden ist und lädt es.

        Args:
            force_download (bool): Erzwingt einen neuen Download.
            **model_load_kwargs: Zusätzliche Argumente, die an torch.load() übergeben werden.

        Returns:
            Das geladene Modell (z.B. ein torch.nn.Module).
            Du MUSST diese Methode an deine spezifische Modell-Ladelogik anpassen!
        """
        model_file_path = self.download_model(force_download=force_download)

        try:
            import torch
            device = model_load_kwargs.get('device', 'cpu')
            map_location = model_load_kwargs.get('map_location', device)

            model_data = torch.load(model_file_path, map_location=map_location)
            logger.info(f"Modell erfolgreich von {model_file_path} geladen.")

            return model_data

        except ImportError:
            logger.error("PyTorch (torch) ist nicht installiert, benötigt für .pth-Dateien.")
            raise
        except Exception as e:
            logger.error(f"Fehler beim Laden der Modell-Datei: {e}")
            raise