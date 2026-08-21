# CTG (NVlabs/CTG) baseline training environment, configured for nuPlan
# via the trajdata_nuplan_diff config already wired in tbsim/configs/registry.py
# (see /home/pete/kshyun/ADCG-Diffusion/baseline_readme.md for how this was
# found -- CTG's own README only documents nuScenes).
#
# Separate image from ADCG-Diffusion's jarcd:latest: CTG pins torch==1.11,
# l5kit==1.5.0, numpy==1.23.4, protobuf==3.20.1 -- an entirely different,
# older dependency stack that would conflict with JARCD's torch==2.3.0 env.

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git wget curl ca-certificates \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 ffmpeg \
        libgdal-dev gdal-bin \
    && rm -rf /var/lib/apt/lists/*

ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p $CONDA_DIR \
    && rm /tmp/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
RUN conda create -n ctg python=3.9 -y
SHELL ["conda", "run", "--no-capture-output", "-n", "ctg", "/bin/bash", "-c"]

WORKDIR /workspace

# setuptools>=81 dropped pkg_resources, which several of these old
# setup.py-only packages assume is importable during their PEP 517 build
# -- same issue and fix as ADCG-Diffusion's own Dockerfile.
RUN echo "setuptools<81" > /tmp/build-constraints.txt
ENV PIP_CONSTRAINT=/tmp/build-constraints.txt
# (PIP_CONSTRAINT deliberately does NOT pin numpy here -- since pip 23.1 it
# applies to isolated PEP 517 build-dependency resolution too, not just the
# main install, so a numpy pin here would collide with l5kit's own
# numpy~=1.19.0 runtime requirement below just as hard as pinning it via a
# normal `pip install numpy==...`. See the numpy pre-install + explicit
# --no-build-isolation a few lines down for how that's actually handled.)

# tbsim pins pytorch-lightning==1.8.3.post0, whose published PyPI wheel has
# malformed metadata (`torch (>=1.9.*)` isn't valid PEP 440) -- pip>=24.1
# refuses to even consider it, exactly as pip's own error message says
# ("Please use pip<24.1 if you need to use this version"). Downgrading pip
# for this whole environment is simpler than re-pinning tbsim's own setup.py.
RUN pip install "pip<24.1"

# Some package pulled in below (transitively) declares an unconstrained
# "numpy" PEP 517 build-time dependency, and pip's isolated build env for
# it resolves that to a numpy old enough to need numpy.distutils, which is
# broken against modern setuptools (`NameError: CCompiler is not defined`).
# It's specifically gym==0.15.4's own build-time numpy requirement -- only
# reproduces once gym is in the mix (confirmed empirically). Installing gym
# on its own with --no-build-isolation (using this pre-installed numpy)
# fixes it without applying --no-build-isolation repo-wide, which causes a
# *different* problem below (some other package's build then can't find
# ITS OWN normally-build-isolated tooling, e.g. opencv-python needing
# scikit-build).
RUN pip install --no-cache-dir numpy==1.23.4 \
    && pip install --no-cache-dir --no-build-isolation gym==0.15.4

# ---- CTG itself ----
# tbsim's setup.py pins numpy==1.23.4, but its own l5kit==1.5.0 dependency
# pins numpy~=1.19.0 -- a real conflict in tbsim's own requirements that
# pip's resolver correctly refuses (tbsim's setup.py even has a comment
# admitting this: "need to manually update numpy version ... due to
# conflict with l5kit's requirement", i.e. the original authors resolved
# this by hand too). Install everything with l5kit's numpy first, then
# force tbsim's actual-wanted numpy version over it, then install tbsim
# itself with --no-deps so pip doesn't re-run the conflicting check.
COPY CTG /workspace/CTG
# l5kit itself doesn't pin a zarr version, so pip's resolver was free to
# pick an ancient pre-wheel zarr release, which falls back to a legacy
# setup.py `fetch_build_eggs` build path that pulls in a broken
# setuptools_scm==10.2.0 (imports a nonexistent `vcs_versioning` module).
# Pinning a modern, wheel-shipping zarr version up front (matching what
# trajdata itself pins later) keeps pip from ever considering that old
# sdist-only release.
# same story for matplotlib: l5kit doesn't pin it either, and pip's
# resolver landed on an ancient 3.0.3 with no wheel for this platform,
# needing a from-source build against freetype/libpng dev headers we don't
# have installed. Pin a modern version (matching trajdata's own "3.5" pin
# below) up front so pip never considers the old one.
# gym is already installed above (see the --no-build-isolation step), but
# l5kit *also* declares its own unconstrained "gym" dependency -- omitting
# it here isn't enough to keep the already-installed version, pip's
# resolver goes hunting through the same broken 0.21-0.26 releases again
# even though 0.15.4 already satisfies the (unconstrained) requirement.
# Pinning it explicitly here too is what actually short-circuits that.
# l5kit (opencv-contrib-python-headless) and gym (opencv-python) both pull
# in unconstrained OpenCV bindings; the specific old release pip's resolver
# picked hard-pins `numpy==1.17.3` in its own [build-system] requires
# (scikit-build-based build) -- numpy 1.17.3 predates Python 3.9 wheels
# entirely, so THAT build falls into the same numpy.distutils/modern-
# setuptools breakage as gym did. Pinning modern OpenCV releases (real
# manylinux wheels, no source build at all) sidesteps it completely.
RUN pip install --no-cache-dir \
        zarr==2.11.0 matplotlib==3.5.3 gym==0.15.4 \
        opencv-python==4.10.0.84 opencv-contrib-python-headless==4.10.0.84 \
        l5kit==1.5.0 pytorch-lightning==1.8.3.post0 wandb torch==1.11 \
        torchvision==0.12.0 pyemd h5py imageio-ffmpeg casadi \
        protobuf==3.20.1 einops==0.6.0 torchtext \
    && pip install --no-cache-dir numpy==1.23.4 \
    && cd CTG && pip install --no-cache-dir --no-deps -e .

# ---- trajdata (CTG's required fork, per its README) ----
COPY ctg_trajdata /workspace/ctg_trajdata
RUN cd ctg_trajdata && pip install --no-cache-dir -r trajdata_requirements.txt && pip install --no-cache-dir -e .

# ---- Pplan (spline-planner) ----
RUN git clone --depth 1 https://github.com/NVlabs/spline-planner.git /workspace/Pplan \
    && cd /workspace/Pplan && pip install --no-cache-dir -e .

# ---- nuplan-devkit (trajdata's nuplan module imports it directly; not an
# extras_require in the trajdata fork above, so installed manually here) ----
# Only its map-stack deps, not the full requirements.txt: that file also
# pins opencv-python<=4.5.1.48 (conflicts with the modern opencv pinned
# above) plus a pile of unrelated tooling (docker, moto, selenium, ray)
# this training path never touches.
# geopandas pulls in numpy>=1.22 unconstrained, and pip's resolver jumped
# straight to numpy 2.0.2 -- a breaking upgrade for everything else in
# this env (torch==1.11 is compiled against the numpy 1.x ABI, plus
# l5kit/scipy/contourpy/tbsim's own exact pins), so force it back down
# immediately after, same pattern as the l5kit numpy fight above.
RUN git clone --depth 1 --branch nuplan-devkit-v1.2.1-release https://github.com/motional/nuplan-devkit.git /workspace/nuplan-devkit \
    && pip install --no-cache-dir geopandas Fiona rasterio pyogrio "Shapely>=2.0.0" aioboto3 aiofiles boto3 urllib3 retry \
    && pip install --no-cache-dir numpy==1.23.4 \
    && cd /workspace/nuplan-devkit && pip install --no-cache-dir --no-deps -e .

# /workspace/Pplan (not /workspace/Pplan/Pplan) needs to be here: the
# upstream NVlabs/spline-planner repo has no Pplan/Pplan/__init__.py, so
# it's an implicit PEP 420 namespace package. The pip editable install's
# finder can't resolve that (it only handles regular packages/modules), so
# `import Pplan.Sampling.tree` silently resolves to an empty broken
# namespace unless the parent directory is on PYTHONPATH for Python's own
# namespace-package discovery to find the subpackages.
ENV PYTHONPATH=/workspace/CTG:/workspace/ctg_trajdata/src:/workspace/nuplan-devkit:/workspace/Pplan:${PYTHONPATH}
ENV OPENBLAS_NUM_THREADS=1
ENV OMP_NUM_THREADS=1

RUN echo "conda activate ctg" >> /root/.bashrc

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "ctg"]
CMD ["/bin/bash"]
