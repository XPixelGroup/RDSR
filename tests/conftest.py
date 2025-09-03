"""
Shared pytest fixtures for the test suite.

This file contains common fixtures that can be used across all test modules.
"""
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock
from typing import Dict, Any, Generator

import pytest
import numpy as np


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for testing."""
    temp_path = tempfile.mkdtemp()
    try:
        yield Path(temp_path)
    finally:
        shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def temp_file(temp_dir: Path) -> Path:
    """Create a temporary file for testing."""
    temp_file_path = temp_dir / "test_file.txt"
    temp_file_path.write_text("test content")
    return temp_file_path


@pytest.fixture
def sample_image() -> np.ndarray:
    """Create a sample image array for testing."""
    return np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)


@pytest.fixture
def sample_grayscale_image() -> np.ndarray:
    """Create a sample grayscale image array for testing."""
    return np.random.randint(0, 255, (64, 64), dtype=np.uint8)


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    model = Mock()
    model.eval = Mock()
    model.forward = Mock(return_value=Mock())
    model.state_dict = Mock(return_value={})
    model.load_state_dict = Mock()
    return model


@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """Create a mock configuration dictionary."""
    return {
        "model": {
            "type": "RRDBNet",
            "num_in_ch": 3,
            "num_out_ch": 3,
            "num_feat": 64,
            "num_block": 23,
            "num_grow_ch": 32,
            "scale": 4
        },
        "datasets": {
            "train": {
                "name": "DIV2K",
                "type": "RealESRGANDataset",
                "dataroot_gt": "/path/to/gt",
                "dataroot_lq": "/path/to/lq"
            }
        },
        "network_g": {
            "type": "RRDBNet",
            "num_in_ch": 3,
            "num_out_ch": 3,
            "num_feat": 64
        },
        "path": {
            "models": "/path/to/models",
            "log": "/path/to/logs"
        }
    }


@pytest.fixture
def mock_device():
    """Mock device for testing."""
    device = Mock()
    device.type = "cpu"
    return device


@pytest.fixture
def sample_tensor_4d():
    """Create a sample 4D tensor for testing (batch, channel, height, width)."""
    import torch
    return torch.randn(1, 3, 32, 32)


@pytest.fixture
def sample_tensor_3d():
    """Create a sample 3D tensor for testing (channel, height, width)."""
    import torch
    return torch.randn(3, 32, 32)


@pytest.fixture
def mock_dataloader():
    """Create a mock dataloader for testing."""
    dataloader = Mock()
    dataloader.__iter__ = Mock(return_value=iter([]))
    dataloader.__len__ = Mock(return_value=0)
    return dataloader


@pytest.fixture
def environment_variables():
    """Fixture to manage environment variables during testing."""
    original_env = os.environ.copy()
    
    def _set_env(**kwargs):
        for key, value in kwargs.items():
            os.environ[key] = str(value)
    
    def _restore_env():
        os.environ.clear()
        os.environ.update(original_env)
    
    yield _set_env
    _restore_env()


@pytest.fixture
def mock_logger():
    """Create a mock logger for testing."""
    logger = Mock()
    logger.info = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    logger.debug = Mock()
    return logger


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test for reproducibility."""
    np.random.seed(42)
    try:
        import torch
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
            torch.cuda.manual_seed_all(42)
    except ImportError:
        pass


@pytest.fixture
def mock_file_system(temp_dir: Path):
    """Create a mock file system structure for testing."""
    # Create mock directories
    (temp_dir / "models").mkdir()
    (temp_dir / "logs").mkdir()
    (temp_dir / "datasets").mkdir()
    (temp_dir / "experiments").mkdir()
    
    # Create some mock files
    (temp_dir / "models" / "pretrained.pth").touch()
    (temp_dir / "datasets" / "image1.png").touch()
    (temp_dir / "datasets" / "image2.png").touch()
    
    return temp_dir


@pytest.fixture
def clean_imports():
    """Clean up imported modules after test to avoid import pollution."""
    import sys
    modules_before = set(sys.modules.keys())
    
    yield
    
    modules_after = set(sys.modules.keys())
    new_modules = modules_after - modules_before
    
    for module in new_modules:
        if module in sys.modules:
            del sys.modules[module]