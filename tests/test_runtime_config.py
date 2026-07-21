import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


CONTRA_ROOT = Path(__file__).resolve().parents[1] / "contra"
sys.path.insert(0, str(CONTRA_ROOT))

import runtime  # noqa: E402
from runtime import build_runtime_config  # noqa: E402


class RuntimeConfigTest(unittest.TestCase):
    def test_auto_uses_local_when_no_server_signal_exists(self):
        with TemporaryDirectory() as temp_dir, patch.object(
            runtime, "SERVER_ELECTRA_PATH", Path(temp_dir) / "missing-model"
        ), patch.dict(
            os.environ,
            {
                "KLJP_MODE": "auto",
                "KLJP_SERVER": "",
                "KLJP_SERVER_HOST_PATTERN": "",
                "SLURM_JOB_ID": "",
                "SLURM_CLUSTER_NAME": "",
                "PBS_JOBID": "",
                "LSB_JOBID": "",
                "KUBERNETES_SERVICE_HOST": "",
            },
            clear=True,
        ):
            project = Path(temp_dir) / "project"
            project.mkdir()
            config = build_runtime_config(project)

        self.assertEqual(config.mode, "local")
        self.assertEqual(config.log_root, project / "logs")
        self.assertTrue(config.allow_model_download)

    def test_explicit_mode_wins_over_auto_detection(self):
        with TemporaryDirectory() as temp_dir, patch.dict(os.environ, {}, clear=True):
            project = Path(temp_dir) / "project"
            project.mkdir()
            (project / ".kljp-server").touch()
            config = build_runtime_config(project, requested_mode="local")

        self.assertEqual(config.mode, "local")
        self.assertEqual(config.reason, "explicit configuration")

    def test_server_model_mount_is_an_auto_detection_signal(self):
        with TemporaryDirectory() as temp_dir, patch.dict(
            os.environ, {"KLJP_MODE": "auto"}, clear=True
        ):
            project = Path(temp_dir) / "project"
            model_path = Path(temp_dir) / "server-model"
            project.mkdir()
            model_path.mkdir()
            with patch.object(runtime, "SERVER_ELECTRA_PATH", model_path):
                config = build_runtime_config(project)

        self.assertEqual(config.mode, "server")
        self.assertIn("server model mount", config.reason)

    def test_server_marker_selects_server_profile(self):
        with TemporaryDirectory() as temp_dir, patch.dict(
            os.environ,
            {"KLJP_MODE": "auto", "KLJP_ALLOW_MODEL_DOWNLOAD": "true"},
            clear=True,
        ):
            project = Path(temp_dir) / "project"
            project.mkdir()
            (project / ".kljp-server").touch()
            config = build_runtime_config(project)

        self.assertEqual(config.mode, "server")
        self.assertEqual(config.data_root, Path("/home/user/KLJP-DATA"))
        self.assertEqual(config.log_root, Path("/home/user/KLJP-logs"))
        self.assertEqual(
            config.electra_path,
            Path("/mntF/XJJ/KLJP-master1/chinese-electra-180g-small-discriminator"),
        )
        self.assertFalse(config.allow_model_download)

    def test_environment_paths_override_profile_defaults(self):
        with TemporaryDirectory() as temp_dir:
            project = Path(temp_dir) / "project"
            project.mkdir()
            overrides = {
                "KLJP_MODE": "local",
                "KLJP_DATA_ROOT": "/tmp/kljp-data",
                "KLJP_LOG_ROOT": "/tmp/kljp-logs",
                "KLJP_ELECTRA_PATH": "/tmp/electra",
            }
            with patch.dict(os.environ, overrides, clear=True):
                config = build_runtime_config(project)

        self.assertEqual(config.data_root, Path("/tmp/kljp-data"))
        self.assertEqual(config.log_root, Path("/tmp/kljp-logs"))
        self.assertEqual(config.electra_path, Path("/tmp/electra"))


if __name__ == "__main__":
    unittest.main()
