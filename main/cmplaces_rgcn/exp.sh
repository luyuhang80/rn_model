#!/usr/bin/env bash
python3 testv2_bare.py --sess bare --ckpt 20
python3 testv2_bare.py --sess bare_md --ckpt 10
python3 testv2_bare.py --sess bare_50k --ckpt 5