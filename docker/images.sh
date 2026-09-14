# Shared image capabilities for build and publish routing.
CPU_BENCHMARKS=(libero libero_pro libero_plus libero_mem robocerebra calvin rlbench duobench robocasa robocasa365 kinetix robomme molmospaces vlabench)
BASE_IMAGES=(base base-cuda base-runtime base-render base-cpu)

BENCHMARKS=(simpler libero libero_pro libero_plus libero_mem robocerebra maniskill2 calvin mikasa_robo vlabench rlbench robotwin robocasa robocasa365 kinetix robomme molmospaces behavior1k duobench robodojo)
DERIVED_BENCHMARKS=(simpler_groot simpler_xvla)
NO_SYSTEM_CUDA_BENCHMARKS=(maniskill2 mikasa_robo robomme molmospaces kinetix duobench simpler libero libero_pro libero_plus libero_mem robocerebra calvin rlbench robocasa robocasa365 vlabench)
NO_REDIST=(rlbench behavior1k robodojo)
