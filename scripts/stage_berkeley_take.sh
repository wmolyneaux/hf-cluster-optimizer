#!/bin/zsh
# Stage the berkeley-usd heroshot render inputs onto the `berkeley-usd-take`
# Modal volume (mounted at /busd by the heroshot_take lane).
#
# MEASURED sizes (du -sm, 2026-08-12): usd 54 MB + textures 240 MB (sky EXR's
# 9 MB inside) + rigged.glb 2 MB + render tools 1 MB + retarget 1.5 MB
# ~= 296 MB total. CPU/network only -- no GPU is billed by any of this.
# NOTE the account's free-storage credit is exactly consumed (billing summary
# 2026-08-11: Volumes $16.25 vs Free Storage -$16.25), so these ~0.3 GB DO
# bill at Modal's volume rate. Pennies/month, but not free -- says so here
# rather than being discovered on an invoice.
#
# Layout on the volume (the lane's trainer asserts every one of these exists
# before any GPU sampling):
#   /usd/...                          the campus USD tree (place_rig hashes it)
#   /textures/...                     incl. textures/_sky/campusSky_*.exr
#   /tools/heroshot/place_rig.py      the render script (script_sha256 input)
#   /tools/render/lut_repair.py       A6 material repair + its LUT
#   /tools/render/material_lut.json
#   /rig/rigged.glb                   the hero rig
#   /shots/<retarget>.json            per-shot retarget solves
set -e
MODAL=/Users/molyneaux/hf-gpu-cluster-optimizer/.venv/bin/modal
BUSD=/Users/molyneaux/Desktop/berkeley-usd
GLB=/Users/molyneaux/golden-rig/out/s1_hairfix/rigged.glb
RET=${1:?usage: stage_berkeley_take.sh /path/to/retarget_<shot>.json}

$MODAL volume create berkeley-usd-take 2>/dev/null || echo "volume exists"
echo "== usd (54 MB) =="
$MODAL volume put --force berkeley-usd-take $BUSD/usd /usd
echo "== textures (240 MB) =="
$MODAL volume put --force berkeley-usd-take $BUSD/textures /textures
echo "== tools =="
$MODAL volume put --force berkeley-usd-take $BUSD/tools/heroshot/place_rig.py /tools/heroshot/place_rig.py
$MODAL volume put --force berkeley-usd-take $BUSD/tools/render/lut_repair.py /tools/render/lut_repair.py
$MODAL volume put --force berkeley-usd-take $BUSD/tools/render/material_lut.json /tools/render/material_lut.json
echo "== rig =="
$MODAL volume put --force berkeley-usd-take $GLB /rig/rigged.glb
echo "== retarget: $RET =="
$MODAL volume put --force berkeley-usd-take $RET /shots/$(basename $RET)
echo "== busd_take.tar (the one workers actually read) =="
# MEASURED on pilot r2: reading the tree through the volume FUSE mount cost
# ~430 s of scene setup per cold container vs ~40 s warm; a single sequential
# tar is the optimized path, and the trainer untars it once to /tmp/busd.
# Rebuild the tar whenever ANY input above changes -- workers prefer the tar
# and would otherwise render yesterday's campus with today's manifest refusing
# to notice (place_rig hashes what it READS, which is the untarred copy).
ST=$(mktemp -d)
mkdir -p $ST/tools/heroshot $ST/tools/render $ST/rig $ST/shots
ln -s $BUSD/usd $ST/usd
ln -s $BUSD/textures $ST/textures
ln -s $BUSD/tools/heroshot/place_rig.py $ST/tools/heroshot/place_rig.py
ln -s $BUSD/tools/render/lut_repair.py $ST/tools/render/lut_repair.py
ln -s $BUSD/tools/render/material_lut.json $ST/tools/render/material_lut.json
ln -s $GLB $ST/rig/rigged.glb
ln -s $RET $ST/shots/$(basename $RET)
tar -chf $ST/busd_take.tar -C $ST usd textures tools rig shots
$MODAL volume put --force berkeley-usd-take $ST/busd_take.tar /busd_take.tar
rm -rf $ST
echo "staged. Verify:"
$MODAL volume ls berkeley-usd-take /
