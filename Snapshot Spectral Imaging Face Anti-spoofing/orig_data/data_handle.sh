
echo "----- start data handle"

if [ -d "./data" ]; then
    rm -rf "./data"
fi
if [ -d "../data_extend" ]; then
    rm -rf "../data_extend"
fi
echo "----- start data sort"
python data_sorting.py

echo "----- start data processing"
python data_processing.py

mv ./data_extend ../

echo "-----  data handle end"
