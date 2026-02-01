SOURCE_DIR="./notebooks/"
TARGET_DIR="./records/"

rm -rf "${TARGET_DIR}"

echo "➡️ Synchronizing .py and .ipynb files including subdirectories, excluding checkpoints..."
rsync -a \
    --exclude='__pycache__/' \
    --exclude='.ipynb_checkpoints/' \
    --include='*/' \
    --include='*.py' \
    --include='*.ipynb' \
    --exclude='*' \
    --prune-empty-dirs \
    "${SOURCE_DIR}" \
    "${TARGET_DIR}"

echo "it is done"