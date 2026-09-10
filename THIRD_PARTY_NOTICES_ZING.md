# Zing inference provenance and third-party notices

This file supplements the repository's Apache License 2.0 and existing notices.
It documents the principal upstream projects used by the Zing-0.5 inference
integration. The corresponding upstream license terms continue to apply to
their respective code and artifacts.

| Component | Upstream | License | Use in this repository |
| --- | --- | --- | --- |
| SGLang | https://github.com/sgl-project/sglang | Apache-2.0 | Serving framework and the base of this fork |
| Zing-0.5 | https://github.com/seedleap/zing-world-model | Apache-2.0 | Public model architecture, checkpoint contract, and reference inference behavior |
| minWM | https://github.com/shengshu-ai/minWM | Apache-2.0 for the Wan implementation used here; see its component-level third-party notices | Causal world-model and action-conditioning reference |
| Wan | https://github.com/Wan-Video/Wan2.1 | Apache-2.0 | Transformer and VAE architecture lineage |
| TAEHV | https://github.com/madebyollin/taehv | MIT | Default local realtime decoder; its checkpoint is downloaded separately |

The Zing integration was modified for SGLang's component loader, realtime
WebSocket API, causal cache management, distributed runtime, and optimized CUDA
execution. It is not an unmodified copy of any single upstream runtime.

Proprietary comparison SDKs, production product integrations, internal cloud
manifests, private benchmark inputs, and copied third-party media assets are not
included in the Zing release surface. The inherited SGLang browser demo still
contains links to upstream-hosted sample images; those images are not stored in
this repository.
