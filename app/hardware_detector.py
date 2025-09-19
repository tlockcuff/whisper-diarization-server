"""Hardware detection and configuration for Whisper Diarization Server."""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import GPUtil
from pynvml import nvmlInit, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetName

logger = logging.getLogger(__name__)


class HardwareDetector:
    """Detect and configure hardware for optimal performance."""

    def __init__(self):
        self.gpu_available = False
        self.gpu_devices = []
        self.cpu_cores = 0
        self.total_memory = 0

    def detect_hardware(self) -> Dict:
        """Detect available hardware and return configuration info."""
        self._detect_gpu()
        self._detect_cpu()
        self._detect_memory()

        return {
            "gpu_available": self.gpu_available,
            "gpu_devices": self.gpu_devices,
            "cpu_cores": self.cpu_cores,
            "total_memory_gb": self.total_memory,
            "recommended_workers": self._calculate_optimal_workers()
        }

    def _detect_gpu(self) -> None:
        """Detect available GPUs."""
        try:
            # Try NVIDIA GPU detection
            nvmlInit()
            device_count = nvmlDeviceGetCount()

            for i in range(device_count):
                try:
                    handle = nvmlDeviceGetHandleByIndex(i)
                    name = nvmlDeviceGetName(handle)
                    memory_info = nvmlDeviceGetMemoryInfo(handle)
                    memory_gb = memory_info.total / (1024**3)  # Convert to GB

                    self.gpu_devices.append({
                        "id": i,
                        "name": name.decode() if isinstance(name, bytes) else str(name),
                        "memory_gb": round(memory_gb, 2)
                    })
                    logger.info(f"Detected GPU {i}: {name} ({memory_gb".2f"}GB)")
                except Exception as e:
                    logger.warning(f"Failed to get info for GPU {i}: {e}")

            self.gpu_available = len(self.gpu_devices) > 0

        except Exception as e:
            logger.info(f"No NVIDIA GPUs detected or nvidia-ml-py3 not available: {e}")

            # Fallback to PyTorch GPU detection
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                for i in range(device_count):
                    try:
                        name = torch.cuda.get_device_name(i)
                        props = torch.cuda.get_device_properties(i)
                        memory_gb = props.total_memory / (1024**3)

                        self.gpu_devices.append({
                            "id": i,
                            "name": name,
                            "memory_gb": round(memory_gb, 2)
                        })
                        logger.info(f"Detected GPU {i}: {name} ({memory_gb".2f"}GB)")
                    except Exception as e:
                        logger.warning(f"Failed to get info for GPU {i}: {e}")

                self.gpu_available = len(self.gpu_devices) > 0

        if not self.gpu_available:
            logger.info("No GPUs detected, will use CPU processing")

    def _detect_cpu(self) -> None:
        """Detect CPU information."""
        try:
            import multiprocessing
            self.cpu_cores = multiprocessing.cpu_count()
            logger.info(f"Detected {self.cpu_cores} CPU cores")
        except Exception as e:
            logger.warning(f"Failed to detect CPU cores: {e}")
            self.cpu_cores = 4  # Default fallback

    def _detect_memory(self) -> None:
        """Detect system memory."""
        try:
            import psutil
            memory_gb = psutil.virtual_memory().total / (1024**3)
            self.total_memory = round(memory_gb, 2)
            logger.info(f"Detected {self.total_memory".2f"}GB system memory")
        except ImportError:
            logger.warning("psutil not available, skipping memory detection")
            self.total_memory = 8.0  # Default fallback
        except Exception as e:
            logger.warning(f"Failed to detect memory: {e}")
            self.total_memory = 8.0  # Default fallback

    def _calculate_optimal_workers(self) -> int:
        """Calculate optimal number of workers based on hardware."""
        if self.gpu_available:
            # For GPU workloads, use fewer workers to avoid memory conflicts
            base_workers = min(len(self.gpu_devices), 2)
        else:
            # For CPU workloads, use more workers
            base_workers = min(self.cpu_cores // 2, 4)

        # Adjust based on memory
        if self.total_memory < 8:
            base_workers = 1
        elif self.total_memory < 16:
            base_workers = min(base_workers, 2)

        logger.info(f"Calculated optimal workers: {base_workers}")
        return base_workers

    def get_optimal_device(self) -> str:
        """Get the optimal device for processing."""
        if self.gpu_available and torch.cuda.is_available():
            return f"cuda:{self.gpu_devices[0]['id']}"
        return "cpu"

    def get_device_info(self) -> Dict:
        """Get comprehensive device information."""
        return {
            "device": self.get_optimal_device(),
            "gpu_count": len(self.gpu_devices),
            "gpu_details": self.gpu_devices,
            "cpu_cores": self.cpu_cores,
            "memory_gb": self.total_memory
        }


# Global hardware detector instance
hardware_detector = HardwareDetector()


def detect_hardware() -> Dict:
    """Detect hardware and return configuration."""
    return hardware_detector.detect_hardware()


def get_optimal_device() -> str:
    """Get the optimal device for processing."""
    return hardware_detector.get_optimal_device()


def get_device_info() -> Dict:
    """Get comprehensive device information."""
    return hardware_detector.get_device_info()