# Model parameters

These four active YAML files were relocated from the repository root without changing their contents. `params.yaml` configures the physical-health regression model; the three outcome-specific files configure the corresponding binary classifiers.

Analysis scripts and notebook code read these files from `./src/analyses/config/` after switching to the project root. The c04 and c07 tuning workflows write updated parameters to the same directory. Preserve filenames and parameter values unless an analysis change explicitly requires otherwise.

CUDA device settings are inherited from the previous environment. Relocation checks verify path resolution and unchanged configuration contents; full model execution on macOS has not been validated.
