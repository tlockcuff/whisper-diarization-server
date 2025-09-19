"""
Hardware detection and compatibility checking for CUDA/GPU support
"""

import torch
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class HardwareDetector:
    """Detect and analyze hardware capabilities for optimal model deployment"""

    def __init__(self):
        self.cuda_available = False
        self.device_props = None
        self.compute_capability = None
        self.gpu_name = None
        self.gpu_memory_gb = None

    def detect_hardware(self) -> Dict[str, any]:
        """Comprehensive hardware detection"""
        info = {
            "cuda_available": False,
            "device_count": 0,
            "primary_device": None,
            "compute_capability": None,
            "gpu_memory_gb": 0,
            "recommendations": [],
            "warnings": []
        }

        try:
            if torch.cuda.is_available():
                info["cuda_available"] = True
                info["device_count"] = torch.cuda.device_count()

                # Analyze primary GPU
                if info["device_count"] > 0:
                    info["primary_device"] = self._analyze_gpu(0)
                    info.update(info["primary_device"])

            else:
                info["warnings"].append("CUDA not available - will use CPU")

        except Exception as e:
            logger.error(f"Hardware detection failed: {e}")
            info["warnings"].append(f"Hardware detection error: {e}")

        return info

    def _analyze_gpu(self, device_idx: int) -> Dict[str, any]:
        """Analyze specific GPU device"""
        try:
            props = torch.cuda.get_device_properties(device_idx)
            compute_cap = f"{props.major}.{props.minor}"

            info = {
                "gpu_name": props.name,
                "compute_capability": compute_cap,
                "gpu_memory_gb": props.total_memory // (1024**3),
                "multiprocessors": props.multi_processor_count,
                "compatibility": self._check_compatibility(compute_cap)
            }

            return info

        except Exception as e:
            logger.error(f"GPU analysis failed for device {device_idx}: {e}")
            return {"error": str(e)}

    def _check_compatibility(self, compute_cap: str) -> Dict[str, any]:
        """Check compatibility with known PyTorch/CUDA limitations"""
        compatibility = {
            "status": "unknown",
            "supported": False,
            "warnings": [],
            "recommendations": []
        }

        # Known compute capabilities and their support status
        supported_cc = {
            "7.0", "7.5",  # V100, Titan V
            "8.0", "8.6", "8.9",  # A100, RTX 30-series, RTX 5060 Ti
            "9.0", "9.0a"  # H100, RTX 40-series
        }

        if compute_cap in supported_cc:
            compatibility["status"] = "supported"
            compatibility["supported"] = True
        else:
            compatibility["status"] = "unsupported"
            compatibility["supported"] = False
            compatibility["warnings"].append(
                f"Compute capability {compute_cap} may not be fully supported"
            )
            compatibility["recommendations"].append(
                "Consider using CPU mode or updating PyTorch/CUDA versions"
            )

        return compatibility

    def get_device_recommendation(self) -> str:
        """Get recommended device for model loading"""
        if self.cuda_available and self.compute_capability:
            # Check if compute capability is known to work
            compat_info = self._check_compatibility(self.compute_capability)
            if compat_info["supported"]:
                return "cuda"
            else:
                logger.warning(f"GPU compatibility issues detected: {compat_info['warnings']}")
                return "cpu"

        return "cpu"

    def print_hardware_info(self):
        """Print comprehensive hardware information"""
        info = self.detect_hardware()

        logger.info("🔍 Hardware Detection Results:")
        logger.info(f"  CUDA Available: {info['cuda_available']}")
        logger.info(f"  Device Count: {info['device_count']}")

        if info.get('primary_device'):
            device = info['primary_device']
            logger.info(f"  Primary GPU: {device.get('gpu_name', 'Unknown')}")
            logger.info(f"  Compute Capability: {device.get('compute_capability', 'Unknown')}")
            logger.info(f"  GPU Memory: {device.get('gpu_memory_gb', 0)}GB")

            compat = device.get('compatibility', {})
            if compat.get('warnings'):
                for warning in compat['warnings']:
                    logger.warning(f"  ⚠️ {warning}")

            if compat.get('recommendations'):
                for rec in compat['recommendations']:
                    logger.info(f"  💡 {rec}")


# Global hardware detector instance
hardware_detector = HardwareDetector()
