# Zing-0.5 examples

See the repository-level [Zing quick start](../../ZING.md) for installation,
model download, server launch, and memory profiles.

`launch_server.sh` starts either the `highmem` profile or the conservative
`ZING_PROFILE=32g` profile. Set `ZING_MODEL_PATH` to the downloaded serving
artifact directory.

`client.py` is intentionally small. It sends one realtime msgpack request,
optionally queues a prompt update, and saves the returned WebP frames without
requiring ffmpeg or a browser.

Examples:

```bash
# T2V with forward motion
python examples/zing_0_5/client.py \
  --prompt "A first-person walk through a misty pine forest" \
  --action w \
  --output outputs/t2v

# I2V with a combined forward/right-camera action
python examples/zing_0_5/client.py \
  --first-frame ./first-frame.png \
  --prompt "Preserve the scene while moving through it" \
  --action w \
  --action l \
  --output outputs/i2v
```
