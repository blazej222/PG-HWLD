$OurDatasetExtractedPath = "../../resources/datasets/dataset-processed/test-images"
$OurDatasetMatPath = "../../resources/datasets/packed/dataset-processed/dataset-processed.mat"
$EMNISTMatPath = "../../resources/datasets/dataset-EMNIST-mat/emnist-letters.mat"


if (-Not (Test-Path -Path "logs")) {
    New-Item -ItemType Directory -Path "logs"
}
if (-Not (Test-Path -Path "logs/test")) {
    New-Item -ItemType Directory -Path "logs/test"
}

Write-Output "Started Testing"

../resources/python/venv/Scripts/activate.ps1
cd ../models/vgg

Write-Output "VGG end result run1 PG-HWLD"
python main.py `
    --test `
    --test_path $OurDatasetExtractedPath `
    --saved_model_path ./saved_models/run1 `
    --cmsuffix end_PG-HWLD_1 `
    | Out-File -FilePath "../../scripts/logs/test/VGG end result run1 PG-HWLD.log"

Write-Output "`nVGG end result run2 PG-HWLD"
python main.py `
    --test `
    --test_path $OurDatasetExtractedPath `
    --saved_model_path ./saved_models/run2 `
    --cmsuffix end_PG-HWLD_2 `
    | Out-File -FilePath "../../scripts/logs/test/VGG end result run2 PG-HWLD.log"

Write-Output "`nVGG end result run3 PG-HWLD"
python main.py `
    --test `
    --test_path $OurDatasetExtractedPath `
    --saved_model_path ./saved_models/run3 `
    --cmsuffix end_PG-HWLD_3 `
    | Out-File -FilePath "../../scripts/logs/test/VGG end result run3 PG-HWLD.log"

Write-Output "`nVGG end result run1 emnist"
python main.py `
    --test `
    --saved_model_path ./saved_models/run1 `
    --cmsuffix end_emnist_1 `
    | Out-File -FilePath "../../scripts/logs/test/VGG end result run1 emnist.log"

Write-Output "`nVGG end result run2 emnist"
python main.py `
    --test `
    --saved_model_path ./saved_models/run2 `
    --cmsuffix end_emnist_2 `
    | Out-File -FilePath "../../scripts/logs/test/VGG end result run2 emnist.log"

Write-Output "`nVGG end result run3 emnist"
python main.py `
    --test `
    --saved_model_path ./saved_models/run3 `
    --cmsuffix end_emnist_3 `
    | Out-File -FilePath "../../scripts/logs/test/VGG end result run3 emnist.log"

Write-Output "`nVGG best result run1 PG-HWLD"
python main.py `
    --test `
    --test_path $OurDatasetExtractedPath `
    --saved_model_path ./saved_models/run1 `
    --model1_filename best_model1.pth `
    --model2_filename best_model2.pth `
    --cmsuffix best_PG-HWLD_1 `
    | Out-File -FilePath "../../scripts/logs/test/VGG best result run1 PG-HWLD.log"

Write-Output "`nVGG best result run2 PG-HWLD"
python main.py `
    --test `
    --test_path $OurDatasetExtractedPath `
    --saved_model_path ./saved_models/run2 `
    --model1_filename best_model1.pth `
    --model2_filename best_model2.pth `
    --cmsuffix best_PG-HWLD_2 `
    | Out-File -FilePath "../../scripts/logs/test/VGG best result run2 PG-HWLD.log"

Write-Output "`nVGG best result run3 PG-HWLD"
python main.py `
    --test `
    --test_path $OurDatasetExtractedPath `
    --saved_model_path ./saved_models/run3 `
    --model1_filename best_model1.pth `
    --model2_filename best_model2.pth `
    --cmsuffix best_PG-HWLD_3 `
    | Out-File -FilePath "../../scripts/logs/test/VGG best result run3 PG-HWLD.log"

Write-Output "`nVGG best result run1 emnist"
python main.py `
    --test `
    --saved_model_path ./saved_models/run1 `
    --model1_filename best_model1.pth `
    --model2_filename best_model2.pth `
    --cmsuffix best_emnist_1 `
    | Out-File -FilePath "../../scripts/logs/test/VGG best result run1 emnist.log"

Write-Output "`nVGG best result run2 emnist"
python main.py `
    --test `
    --saved_model_path ./saved_models/run2 `
    --model1_filename best_model1.pth `
    --model2_filename best_model2.pth `
    --cmsuffix best_emnist_2 `
    | Out-File -FilePath "../../scripts/logs/test/VGG best result run2 emnist.log"

Write-Output "`nVGG best result run3 emnist"
python main.py `
    --test `
    --saved_model_path ./saved_models/run3 `
    --model1_filename best_model1.pth `
    --model2_filename best_model2.pth `
    --cmsuffix best_emnist_3 `
    | Out-File -FilePath "../../scripts/logs/test/VGG best result run3 emnist.log"

cd ../WaveMix
Write-Output "`nWaveMix best result run1 PG-HWLD"
python main.py `
    --test `
    --test_path $OurDatasetExtractedPath `
    --saved_model_path ./saved_models/run1 `
    --model_filename model.pth `
    --cmsuffix PG-HWLD_1 `
    | Out-File -FilePath "../../scripts/logs/test/WaveMix best result run1 PG-HWLD.log"

Write-Output "`nWaveMix best result run2 PG-HWLD"
python main.py `
    --test `
    --test_path $OurDatasetExtractedPath `
    --saved_model_path ./saved_models/run2 `
    --model_filename model.pth `
    --cmsuffix PG-HWLD_2 `
    | Out-File -FilePath "../../scripts/logs/test/WaveMix best result run2 PG-HWLD.log"

Write-Output "`nWaveMix best result run3 PG-HWLD"
python main.py `
    --test `
    --test_path $OurDatasetExtractedPath `
    --saved_model_path ./saved_models/run3 `
    --model_filename model.pth `
    --cmsuffix PG-HWLD_3 `
    | Out-File -FilePath "../../scripts/logs/test/WaveMix best result run3 PG-HWLD.log"

Write-Output "`nWaveMix best result run1 emnist"
python main.py `
    --test `
    --saved_model_path ./saved_models/run1 `
    --model_filename model.pth `
    --cmsuffix emnist_1 `
    | Out-File -FilePath "../../scripts/logs/test/WaveMix best result run1 emnist.log"

Write-Output "`nWaveMix best result run2 emnist"
python main.py `
    --test `
    --saved_model_path ./saved_models/run2 `
    --model_filename model.pth `
    --cmsuffix emnist_2 `
    | Out-File -FilePath "../../scripts/logs/test/WaveMix best result run2 emnist.log"

Write-Output "`nWaveMix best result run3 emnist"
python main.py `
    --test `
    --saved_model_path ./saved_models/run3 `
    --model_filename model.pth `
    --cmsuffix emnist_3 `
    | Out-File -FilePath "../../scripts/logs/test/WaveMix best result run3 emnist.log"

cd ../..
deactivate
resources/python/venv-3.6.8/Scripts/activate.ps1
cd models/TextCaps

Write-Output "`nTextcaps end result run1 PG-HWLD.log"
python textcaps_emnist_bal.py `
    --test `
    --weights ./saved_models/emnist_letters_5600_run1/trained_model.h5 `
    --test_path $OurDatasetMatPath `
    --cmsuffix PG-HWLD_1 `
    | Out-File -FilePath "../../scripts/logs/test/Textcaps end result run1 PG-HWLD.log"

Write-Output "`nTextcaps end result run2 PG-HWLD"
python textcaps_emnist_bal.py `
    --test `
    --weights ./saved_models/emnist_letters_5600_run2/trained_model.h5 `
    --test_path $OurDatasetMatPath `
    --cmsuffix PG-HWLD_2 `
    | Out-File -FilePath "../../scripts/logs/test/Textcaps end result run2 PG-HWLD.log"

Write-Output "`nTextcaps end result run3 PG-HWLD"
python textcaps_emnist_bal.py `
    --test `
    --weights ./saved_models/emnist_letters_5600_run3/trained_model.h5 `
    --test_path $OurDatasetMatPath `
    --cmsuffix PG-HWLD_3 `
    | Out-File -FilePath "../../scripts/logs/test/Textcaps end result run3 PG-HWLD.log"

Write-Output "`nTextcaps end result run1 emnist"
python textcaps_emnist_bal.py `
    --test `
    --weights ./saved_models/emnist_letters_5600_run1/trained_model.h5 `
    --test_path $EMNISTMatPath `
    --cmsuffix emnist_1 `
    | Out-File -FilePath "../../scripts/logs/test/Textcaps end result run1 emnist.log"

Write-Output "`nTextcaps end result run2 emnist"
python textcaps_emnist_bal.py `
    --test `
    --weights ./saved_models/emnist_letters_5600_run2/trained_model.h5 `
    --test_path $EMNISTMatPath `
    --cmsuffix emnist_2 `
    | Out-File -FilePath "../../scripts/logs/test/Textcaps end result run2 emnist.log"

Write-Output "`nTextcaps end result run3 emnist"
python textcaps_emnist_bal.py `
    --test `
    --weights ./saved_models/emnist_letters_5600_run3/trained_model.h5 `
    --test_path $EMNISTMatPath `
    --cmsuffix emnist_3 `
    | Out-File -FilePath "../../scripts/logs/test/Textcaps end result run3 emnist.log"

deactivate
cd ../../scripts