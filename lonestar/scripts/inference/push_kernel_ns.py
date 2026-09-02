#!/usr/bin/env python3
"""Push the NS submission notebook with machine_shape=NvidiaTeslaT4 (runs on TACC, rsna_env)."""
import json
from kagglesdk import KaggleClient, KaggleEnv
from kagglesdk.kernels.types.kernels_api_service import ApiSaveKernelRequest

creds = json.load(open("/home1/07880/devansh/.kaggle/kaggle.json"))
client = KaggleClient(env=KaggleEnv.PROD, verbose=False,
                      username=creds["username"], password=creds["key"])
req = ApiSaveKernelRequest()
req.slug = "devansh26/rsna-knee-r2-inference"
req.new_title = "rsna-knee-r2-inference"
req.text = open("/work/07880/devansh/lonestar/rsna_knee/notebooks/kaggle_submission_ns.ipynb").read()
req.language = "python"
req.kernel_type = "notebook"
req.is_private = True
req.enable_gpu = True
req.enable_internet = False
req.machine_shape = "NvidiaTeslaT4"
req.dataset_data_sources = ["devansh26/rsna-knee-weights-r2"]
req.competition_data_sources = ["rsna-knee-abnormality-detection"]
resp = client.kernels.kernels_api_client.save_kernel(req)
print("ref:", resp.ref, "| version:", getattr(resp, "version_number", "?"))
print("error:", getattr(resp, "error", "") or "none")
