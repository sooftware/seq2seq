HIDDEN_DIM=256
NUM_HEADS=4
D_FF=512
NUM_ENCODER_LAYERS=3
NUM_DECODER_LAYERS=2
DROPOUT_P=0.3
RNN_TYPE='lstm'

# shellcheck disable=SC2164
cd toy_problem
python toy.py --hidden_dim $HIDDEN_DIM --num_heads $NUM_HEADS --d_ff $D_FF \
--num_encoder_layers $NUM_ENCODER_LAYERS --num_decoder_layers $NUM_DECODER_LAYERS \
--dropout_p $DROPOUT_P --rnn_type $RNN_TYPE
# shellcheck disable=SC2103
cd ..
