import json
import sys

import av


def _rate_to_string(rate) -> str:
    if rate is None:
        return "30/1"
    numerator = getattr(rate, "numerator", None)
    denominator = getattr(rate, "denominator", None)
    if numerator is not None and denominator:
        return f"{numerator}/{denominator}"
    return f"{float(rate):.6f}/1"


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    input_path = next((arg for arg in reversed(args) if not arg.startswith("-")), None)
    if input_path is None:
        return 1

    container = av.open(input_path)
    streams = []
    for stream in container.streams:
        codec_ctx = stream.codec_context
        if stream.type == "video":
            streams.append({
                "codec_type": "video",
                "codec_name": codec_ctx.name or "h264",
                "width": stream.width or codec_ctx.width or 0,
                "height": stream.height or codec_ctx.height or 0,
                "r_frame_rate": _rate_to_string(stream.average_rate or stream.base_rate or stream.guessed_rate),
                "bit_rate": str(stream.bit_rate or 0),
                "pix_fmt": codec_ctx.pix_fmt or "yuv420p",
            })
        elif stream.type == "audio":
            streams.append({
                "codec_type": "audio",
                "codec_name": codec_ctx.name or "aac",
                "sample_rate": str(codec_ctx.sample_rate or 44100),
                "channels": codec_ctx.channels or 2,
                "bit_rate": str(stream.bit_rate or 128000),
            })

    duration = 0.0
    if container.duration is not None:
        duration = float(container.duration / av.time_base)

    payload = {
        "streams": streams,
        "format": {
            "duration": f"{duration:.6f}",
            "bit_rate": str(container.bit_rate or 0),
        },
    }
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
