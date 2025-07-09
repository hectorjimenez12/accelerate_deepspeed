

tests:

- python examples/example_5_fsdp_cnn/fsdp_memory_test.py --batch-size 1024 --width 4 (40% two gpus)
- python examples/example_5_fsdp_cnn/fsdp_memory_test.py --batch-size 1024 --width 8 (90% two gpus)
- python fsdp_memory_test.py --batch-size 1024 --width 4 --amp
-  python examples/example_5_fsdp_cnn/fsdp_memory_test.py --batch-size 1024 --width 8 --amp --checkpoint (14 % each GPU)
