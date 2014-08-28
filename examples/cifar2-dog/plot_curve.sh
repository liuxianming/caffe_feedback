#!/usr/bin/env sh

TOOLS=../../tools/extra

#python $TOOLS/plot_training_log.py 1 curves_ff.png train_ff.log
#python $TOOLS/plot_training_log.py 1 curves_fb.png train_fb.log
#python $TOOLS/plot_training_log.py 1 curves.png train_ff.log train_fb.log
python $TOOLS/plot_training_log_multiple.py 40 curves_acc.png train_ff.log train_fb.log
python $TOOLS/plot_training_log_multiple.py 82 curves_loss.png train_ff.log train_fb.log
sz curves_acc.png curves_loss.png
rm curves_acc.png curves_loss.png

echo "Done."
