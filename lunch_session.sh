#!/bin/bash
module load python/3.9 cuda cudnn
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index numpy -q
pip install --no-index torch -q
pip install --no-index tqdm -q
pip install --no-index pandas -q
pip install --no-index json -q
pip install --no-index matplotlib -q
pip install --no-index time -q
pip install --no-index transformers -q
pip install --no-index SentencePiece -q
pip install --no-index torchtext -q
echo "installations done"

python /home/sreyd/projects/def-azouaq/sreyd/NMT/scripts/main.py $@
