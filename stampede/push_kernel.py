#!/usr/bin/env python3
"""Push the submission kernel with machine_shape=NvidiaTeslaT4.

The plain `kaggle kernels push` CLI cannot set the accelerator and lands on
P100, which this competition forbids. Always push with this script.
Requires KAGGLE_API_TOKEN in the env (source env.sh).
"""
import sys

sys.path.insert(0, "/scratch/09059/xliaoyi/kaggle/sdk128")
from kagglesdk import KaggleClient, KaggleEnv
from kagglesdk.kernels.types.kernels_api_service import ApiSaveKernelRequest

SCRIPT = "/scratch/09059/xliaoyi/kaggle/kernel_submit_meta/rsna-submit-v1.py"

client = KaggleClient(env=KaggleEnv.PROD, verbose=False,
                      username="liaoyixu", password="x")
req = ApiSaveKernelRequest()
req.slug = "liaoyixu/rsna-submit-v1"
req.new_title = "rsna-submit-v1"
req.text = open(SCRIPT).read()
req.language = "python"
req.kernel_type = "script"
req.is_private = True
req.enable_gpu = True
req.enable_internet = False
req.machine_shape = "NvidiaTeslaT4"
req.dataset_data_sources = ["liaoyixu/rsna-knee-code", "liaoyixu/rsna-knee-weights",
                            "liaoyixu/rsna-torch-cu118"]
req.competition_data_sources = ["rsna-knee-abnormality-detection"]
resp = client.kernels.kernels_api_client.save_kernel(req)
print("ref:", resp.ref, "| version:", getattr(resp, "version_number", "?"))
print("error:", getattr(resp, "error", "") or "none")
