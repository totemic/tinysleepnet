inference:
	python -u predict_raw.py --config_file config/sleepedf.py --model_dir out_sleepedf/train --use-best \
		2>&1| tee $@.log
