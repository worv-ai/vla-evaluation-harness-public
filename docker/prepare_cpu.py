"""Replace accelerator wheels before exporting a CPU runtime layer."""

import importlib.metadata as metadata
import subprocess
import sys


def main():
    installed = {d.metadata["Name"].lower().replace("_", "-"): d.version for d in metadata.distributions()}
    wheels = [
        name + "==" + installed[name].split("+")[0] + "+cpu"
        for name in ("torch", "torchvision", "torchaudio")
        if name in installed and not installed[name].endswith("+cpu")
    ]
    uv = ["uv", "pip"]
    if wheels:
        subprocess.run(
            uv
            + [
                "install",
                "--python",
                sys.executable,
                "--no-cache",
                "--no-deps",
                "--reinstall",
                "--torch-backend",
                "cpu",
                "--index-url",
                "https://download.pytorch.org/whl/cpu",
                *wheels,
            ],
            check=True,
        )
    accelerator = [
        name
        for name in installed
        if name.startswith(("nvidia-", "jax-cuda", "cupy-cuda", "cuda-")) or name == "triton"
    ]
    if accelerator:
        subprocess.run(uv + ["uninstall", "--python", sys.executable, *accelerator], check=True)


if __name__ == "__main__":
    main()
