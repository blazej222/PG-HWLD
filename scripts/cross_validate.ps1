Write-Output "Dataset division sequence"

$datasetRoot = "../resources/datasets/dataset-processed/test-images"
$outputRoot = "../resources/datasets/temp/subsets"
$trainDir = "../resources/datasets/temp/train-images"
$testDir = "../resources/datasets/temp/test-images"

function Shuffle-Array {
    param([Object[]]$array)
    $array | Sort-Object { Get-Random }
}

function Dummy-Function {
    param([string]$trainPath, [string]$testPath, [int]$fold)
    Write-Output "Processing fold $fold"

    

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
            Copy-Item -Recurse -Path $sourceDir -Destination $testDir
        } else {
            Copy-Item -Recurse -Path $sourceDir -Destination $trainDir
        }
    }

    Dummy-Function -trainPath $trainDir -testPath $testDir
}

Read-Host -Prompt "Press any key to continue"