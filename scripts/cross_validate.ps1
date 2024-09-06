Write-Output "Dataset division sequence"

$datasetRoot = "../resources/datasets/dataset-processed/test-images"
$outputRoot = "../resources/datasets/temp/subsets"
$trainDir = "../resources/datasets/temp/train-images"
$testDir = "../resources/datasets/temp/test-images"
$datasetMatPath = "../resources/datasets/temp/mat"
$tempDir = "../resources/datasets/temp"

function Shuffle-Array {
    param([Object[]]$array)
    $array | Sort-Object { Get-Random }
}

function Perform-Training {
    param([string]$trainPath, [string]$testPath, [int]$fold)

    ../resources/python/venv/Scripts/activate.ps1

    $ErrorActionPreference="SilentlyContinue"
    Stop-Transcript | out-null
    $ErrorActionPreference = "Continue"

    Write-Output "Processing vgg"
    python ../models/vgg/main.py --train_path "$trainDir" --test_path "$testDir" --model1_filename "vgg $fold model1.pth" --model2_filename "spinalnet $fold model2.pth" | Out-File -FilePath "logs/cross_validate/vgg $fold.txt"

    Write-Output "Processing WaveMix"
    python ../models/WaveMix/main.py --train_path "$trainDir" --test_path "$testDir" --model_filename "WaveMix $fold model.pth" | Out-File -FilePath "logs/cross_validate/WaveMix $fold.txt"

    Write-Output "Packing dataset"
    python ../utils/DataSetPacker/main.py --source "$tempDir" --destination "$datasetMatPath/" --filename "packed_$fold.mat"

    deactivate

    ../resources/python/venv-3.6.8/Scripts/activate.ps1
    Write-Output "Processing TextCaps"

    $env:PYTHONIOENCODING="utf-8"
    python ../models/TextCaps/textcaps_emnist_bal.py --save_dir "saved_models/TextCaps/$fold" --train_path "$datasetMatPath/packed_$fold.mat" --test_path "$datasetMatPath/packed_$fold.mat"

    deactivate
}

if(-not (Test-Path -Path $outputRoot)) {
            New-Item -ItemType Directory -Path $outputRoot -Force
}

$subsets = 1..10 | ForEach-Object {
    $subsetDir = Join-Path $outputRoot "Subset$_"
    New-Item -ItemType Directory -Path $subsetDir -Force
}

Get-ChildItem -Path $datasetRoot -Directory | ForEach-Object {
    $classDir = $_

    $files = Get-ChildItem -Path $classDir.FullName -File

    $shuffledFiles = Shuffle-Array -array $files

    for ($i = 0; $i -lt $shuffledFiles.Count; $i++) {
        $subsetIndex = $i % 10

        $destinationDir = Join-Path $subsets[$subsetIndex] $classDir.Name

        if(-not (Test-Path -Path $destinationDir)) {
            New-Item -ItemType Directory -Path $destinationDir -Force
        }

        Copy-Item -Path $shuffledFiles[$i].FullName -Destination $destinationDir
    }
}

Write-Output "Finished subset creation"

for ($testIndex = 0; $testIndex -lt 10; $testIndex++) {
    Write-Output "Processing fold $($testIndex + 1)"

    if (Test-Path -Path $trainDir) {
        Remove-Item -Recurse -Force $trainDir
    }
    if (Test-Path -Path $testDir) {
        Remove-Item -Recurse -Force $testDir
    }

    New-Item -ItemType Directory -Path $trainDir -Force
    New-Item -ItemType Directory -Path $testDir -Force

    for ($i = 0; $i -lt 10; $i++) {
        $sourceDir = $subsets[$i]

        if ($i -eq $testIndex) {
            Get-ChildItem -Path $sourceDir | ForEach-Object {
                Copy-Item -Force -Recurse -Path $_.FullName -Destination $testDir
            }     
        } else {
            Get-ChildItem -Path $sourceDir | ForEach-Object {
                Copy-Item -Force -Recurse -Path $_.FullName -Destination $trainDir
            }
        }
    }

    Perform-Training -trainPath $trainDir -testPath $testDir -fold $testIndex
}

Read-Host -Prompt "Process finished, press any key close"