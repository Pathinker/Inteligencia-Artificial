if [ ! -d "./dataset" ]; then
    start=$(date +%s)

    mkdir -p ./dataset
    curl -L -o ./dataset/open-sprayer-images.zip \
    https://www.kaggle.com/api/v1/datasets/download/gavinarmstrong/open-sprayer-images

    unzip ./dataset/open-sprayer-images.zip -d ./dataset
    rm ./dataset/open-sprayer-images.zip

    rm -rf ./dataset/Docknet/Docknet
    mv ./dataset/Docknet/* ./dataset
    rm -rf ./dataset/Docknet

    end=$(date +%s)
    duration=$((end - start))

    echo "Dataset unpacked suscefully in $duration seconds."
fi