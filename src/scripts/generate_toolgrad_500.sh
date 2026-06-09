seq 1 500 | xargs -n 1 -P 2 -I {} python src/generate_toolgrad_data.py \
    --cfg examples/configs/gemini-2.5-lite-vertex.gin \
    --iter 10 \
    --num_apis 50 \
    --output_dir $HOME/data/toolgrad-data/datasets/toolgrad_500_gemini_2.5_flash_lite/ \
    --seed {}