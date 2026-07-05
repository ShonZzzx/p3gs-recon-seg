# ROI RaDe-GS + GOF scripts

This folder prepares plant-centered ROI datasets and runs ROI-RaDe-GS plus
ROI-GOF with the same split and resolution.

Default server paths:

- data: `/root/autodl-tmp/ml_course/dataGS`
- ROI data: `/root/autodl-tmp/ml_course/dataGS_roi`
- outputs: `/root/autodl-tmp/ml_course/compare_roi_r2_30k`
- RaDe-GS repo: `/root/autodl-tmp/ml_course/repos/RaDe-GS`
- GOF repo: `/root/autodl-tmp/ml_course/repos/gaussian-opacity-fields`

Main commands:

```bash
cd /root/autodl-tmp/ml_course/scripts/roi_fusion
bash check_roi_ready.sh
bash run_prepare_roi.sh
bash run_roi_rade_gof_queue.sh
```

After renting a GPU server with CUDA toolkit:

```bash
cd /root/autodl-tmp/ml_course/scripts/roi_fusion
bash setup_after_gpu.sh
RADE_PYTHON=/root/autodl-tmp/ml_course/envs/radegs/bin/python \
GOF_PYTHON=/root/autodl-tmp/ml_course/envs/gof/bin/python \
bash run_roi_rade_gof_queue.sh
```

Useful overrides:

```bash
PLANTS="plant_002 plant_013 plant_016 plant_019" RESOLUTION=2 ITERATIONS=30000 bash run_roi_rade_gof_queue.sh
MODE=parallel bash run_roi_rade_gof_queue.sh
OVERWRITE=1 bash run_prepare_roi.sh
ROI_DRIVER=colmap bash run_prepare_roi.sh
```

The default ROI crop uses `ROI_DRIVER=hybrid`: green plant foreground first,
registered COLMAP 2D points as fallback. The original full images are still
used for COLMAP poses; the script crops images and writes adjusted
`cameras.bin` and `images.bin` under `sparse/0`.
