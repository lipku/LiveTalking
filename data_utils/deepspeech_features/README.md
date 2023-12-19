# Routines for DeepSpeech features processing
Several routines for [DeepSpeech](https://github.com/mozilla/DeepSpeech) features processing, like speech features generation for [VOCA](https://github.com/TimoBolkart/voca) model.

## Installation

```
pip3 install -r requirements.txt
```

## Usage

Generate wav files:
```
python3 extract_wav.py --in-video=<you_data_dir>
```

Generate files with DeepSpeech features:
```
python3 extract_ds_features.py --input=<you_data_dir>
```
