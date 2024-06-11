$OurDatasetExtractedPath = "../../resources/datasets/dataset-processed"
$OurDatasetMatPath = "../../resources/datasets/packed/dataset-processed/"

Write-Output "Packing dataset"
../resources/python/venv/Scripts/activate.ps1
Set-Location ../utils/DataSetPacker

python main.py --source $OurDatasetExtractedPath --destination $OurDatasetMatPath --filename dataset-processed.mat
deactivate
Set-Location ../../scripts