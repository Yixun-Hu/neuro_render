eval "$(conda shell.bash hook)"
source "$CONDA_PREFIX/etc/profile.d/conda.sh"

ENV_NAME='gs2d'

conda create -n $ENV_NAME python=3.11 -y
conda activate $ENV_NAME

# !!! Ensure correct environment is actually activated !!! #
if [ "$CONDA_DEFAULT_ENV" = "$ENV_NAME" ]; then
    echo "CONDA_DEFAULT_ENV is $ENV_NAME"
else
    echo "Error: CONDA_DEFAULT_ENV is not $ENV_NAME"
    exit 1
fi


pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
conda install -c conda-forge gcc_linux-64=11 gxx_linux-64=11 ninja -y
conda install -c nvidia cuda-toolkit=12.8 -y
export CC=$(which x86_64-conda-linux-gnu-gcc)
export CXX=$(which x86_64-conda-linux-gnu-g++)
export CUDA_HOME="$CONDA_PREFIX"
export CUDACXX="$CONDA_PREFIX/bin/nvcc"
export CUDA_PATH="$CUDA_HOME"
export CPATH="$CONDA_PREFIX/targets/x86_64-linux/include:$CPATH"
export LIBRARY_PATH="$CONDA_PREFIX/targets/x86_64-linux/lib:$LIBRARY_PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/targets/x86_64-linux/lib:$LD_LIBRARY_PATH"
conda env update -f environment.yml

pip install -q numpy==1.26.4 opencv-python==4.9.0.80
pip install gdown
pip install --force-reinstall --no-binary :all: pillow==10.2.0
