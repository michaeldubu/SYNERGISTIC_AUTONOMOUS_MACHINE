To use this script:

1. **Save your Claude and DeepSeek data** in a folder (e.g., `./input_data/`)
2. **Run the setup script**: `python setup_sam.py --data_dir ./input_data/`
3. **Start training**: `python run.py --mode train --train_data ./data/processed/train.json --eval_data ./data/processed/eval.json`

The script handles:
- Creating all necessary directories
- Converting various data formats (text, JSON, JSONL) to SAM's format
- Creating an appropriate initial configuration for your Titan X Pascal
- Splitting data into training and evaluation sets

It automatically recognizes common formats like:
- Claude dialogues
- DeepSeek instruction/output formats
- Plain text documents
- JSONL records