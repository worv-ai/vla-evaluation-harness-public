"""``vla-eval recording-daemon`` + ``vla-eval end-eval`` subcommands."""

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

    end_eval_parser = sub.add_parser(
        "end-eval",
        help="Send EVAL_END to a running recording daemon (used by run_sharded.sh after all shards finish)",
    )
    end_eval_parser.add_argument(
        "--recording-daemon-url",
        required=True,
        help="WebSocket URL of the running recording daemon (e.g. ws://127.0.0.1:9001)",
    )
    end_eval_parser.add_argument("--eval-id", required=True, help="The eval id to close")
    end_eval_parser.add_argument("--verbose", "-v", action="store_true")
    end_eval_parser.set_defaults(func=cmd_end_eval)


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


def cmd_end_eval(args: argparse.Namespace) -> None:
    """Push EVAL_END to a daemon — used by external coordinators (run_sharded.sh)
    that need to close an eval whose RESULTs came from N shard processes."""
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    from vla_eval.recording_daemon.client import RecordingClient

    client = RecordingClient(args.recording_daemon_url)
    try:
        client.eval_end(args.eval_id)
    finally:
        client.close()


async def _run(cfg: RecordingDaemonConfig) -> None:
    daemon = RecordingDaemon(cfg)
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda: asyncio.ensure_future(daemon.stop()))
    await daemon.start()
