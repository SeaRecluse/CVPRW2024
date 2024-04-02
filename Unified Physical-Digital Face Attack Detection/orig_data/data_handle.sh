
echo "----- start data handle"

echo "----- start data sort"
python data_sorting.py

echo "----- start data processing"
python data_processing.py

if [ -d "../data" ]; then
    rm -rf "../data"
fi
mv ./data ../

echo "-----  data handle end"
