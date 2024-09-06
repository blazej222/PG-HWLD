Write-Output "Started Testing cross validation"

../resources/python/venv/Scripts/activate.ps1
cd ../models/vgg

for ($i=0;$i-le 9;$i++)
{
    Write-Output "`nVGG+Spinal best $i"
    python main.py `
        --test `
        --test_path "../../resources/datasets/temp/subsets/Subset$($i+1)" `
        --saved_model_path "../../scripts/saved_models/" `
        --model1_filename "best_vgg $i model1.pth" `
        --model2_filename "best_spinalnet $i model2.pth" `
        --do_not_rotate_images `
        | Out-File -FilePath "../../scripts/logs/test_cross_validate/VGG+spinal best subset $($i+1).log"

#    Write-Output "nVGG+Spinal end $i"
#    python main.py `
#        --test `
#        --test_path "../../resources/datasets/temp/subsets/Subset$($i+1)" `
#        --saved_model_path "../../scripts/saved_models/" `
#        --model1_filename "vgg $i model1.pth" `
#        --model2_filename "spinalnet $i model2.pth" `
#        | Out-File -FilePath "../../scripts/logs/test_cross_validate/VGG+spinal end subset $($i+1).log"
}

cd ../WaveMix
for ($i=0;$i-le 9;$i++)
{
    Write-Output "`nWaveMix $i"
    python main.py `
        --test `
        --test_path "../../resources/datasets/temp/subsets/Subset$($i+1)" `
        --saved_model_path "../../scripts/saved_models/" `
        --model_filename "WaveMix $i model.pth" `
        --do_not_rotate_images `
        | Out-File -FilePath "../../scripts/logs/test_cross_validate/WaveMix $($i+1).log"
}

cd ../..
deactivate
resources/python/venv-3.6.8/Scripts/activate.ps1
cd models/TextCaps

for ($i=0;$i-le 9;$i++)
{
    Write-Output "`nTextcaps $i"
    python textcaps_emnist_bal.py `
        --test `
        --weights "../../scripts/saved_models/TextCaps/$i/trained_model.h5" `
        --test_path "../../resources/datasets/temp/mat/packed_$i.mat" `
        | Out-File -FilePath "../../scripts/logs/test_cross_validate/TextCaps $($i+1).log"
}

deactivate
cd ../../scripts