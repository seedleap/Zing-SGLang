# Zing-0.5 release checklist

Use this checklist for both the SGLang source fork and the converted ModelScope
artifact.

## Source repository

- Base the public branch on the selected public SGLang revision. Never publish
  development branch ancestry that contains private deployment history.
- Keep only model inference, generic realtime serving, examples, tests, and
  public provenance. Exclude production manifests, credentials, private
  benchmark inputs, proprietary SDKs, and copied media.
- Run Ruff, Python compilation, `git diff --check`, and secret scanning on the
  exact public tree.
- Run the Zing unit tests and one T2V plus one I2V smoke test on Linux/CUDA.
- Record the tested SGLang commit, CUDA/PyTorch versions, GPU name, GPU memory,
  launch command, cache profile, and output hashes.

## Model artifact

- Convert only from the public `seedleap/Zing-0.5` release.
- Copy `modelcards/zing-0.5-sglang/README.md` to the artifact root and include
  the repository's Apache-2.0 `LICENSE` before checksumming.
- Export the exact clean public source revision into `sglang-runtime/` without
  its `.git` directory. Materialize tracked symlinks as regular files so the
  complete ModelScope artifact remains portable and passes the symlink gate.
- Do not use `--link-donor` for a publishable artifact.
- Run:

  ```bash
  python -m sglang.multimodal_gen.tools.verify_zing_artifact \
    ./models/Zing-0.5-SGLang \
    --write-sha256
  ```

- Verify that `model_index.json` names `ZingCausalDMDPipeline`, the scheduler is
  `null`, every shard in the index exists, and no symlink or private path remains.
- Create `seedleap/Zing-0.5-SGLang` as **private**, upload with a resumable
  large-file mechanism, and verify every remote file and checksum.
- Keep the model private until the source repository, license review, CUDA smoke
  tests, and final release approval are complete.
