"""``vla-eval recording-daemon`` subcommand."""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
from typing import Any

from vla_eval.recording_daemon.config import RecordingDaemonConfig
from vla_eval.recording_daemon.daemon import RecordingDaemon

logger = logging.getLogger(__name__)


def add_subparser(sub: argparse._SubParsersAction[Any]) -> None:
    parser = sub.add_parser(
        "recording-daemon",
        help="Run the standalone recording daemon (sidecar to model server + shards)",
    )
    parser.add_argument("--bind", default="ws://0.0.0.0:9001", help="WebSocket bind URL (default ws://0.0.0.0:9001)")
    parser.add_argument(
        "--out-dir",
        default="./recording-out",
        help="Fallback directory for aged-out / shutdown-flushed buckets without a known path",
    )
    parser.add_argument(
        "--idle-timeout-sec",
        type=float,
        default=600.0,
        help="Age-out window for buckets/evals without RESULT/EVAL_END",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.set_defaults(func=cmd_recording_daemon)


def cmd_recording_daemon(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    cfg = RecordingDaemonConfig(
        bind=args.bind,
        out_dir=args.out_dir,
        idle_timeout_sec=args.idle_timeout_sec,
    )
    asyncio.run(_run(cfg))


async def _run(cfg: RecordingDaemonConfig) -> None:
    daemon = RecordingDaemon(cfg)
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda: asyncio.ensure_future(daemon.stop()))
    await daemon.start()
