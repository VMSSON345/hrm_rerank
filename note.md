Tổng nợ phải trả của công ty trong năm 2025 là bao nhiêu? Hãy liệt kê rõ nợ ngắn hạn và nợ dài hạn để kiểm tra. Tổng tài sản tính đến ngày 31/12/2025 có bằng Tổng nguồn vốn không? Nếu lệch thì chênh lệch là bao nhiêu? Tính hệ số thanh toán hiện hành (Current Ratio) năm 2025 và cho biết doanh nghiệp có đủ khả năng trả nợ ngắn hạn không? Tổng nợ vay tài chính chịu lãi (gồm vay ngắn hạn và dài hạn) chiếm bao nhiêu phần trăm trên Tổng nguồn vốn?



Đọc strategy.md và tiếp tục từ cuối S4.1. Phase D đang khóa vì overall accepted precision 88,47% < 90%. Chỉ phân tích false accept development-only của SELECT/ADD theo proof obligation và lỗi cấu trúc tổng quát. Không đọc validation/blind, không fine-tune và không thay threshold.

(base) PS D:\reasoning_model_V4\reasoning_model> wsl bash -lc "cd /mnt/d/reasoning_model_V4/reasoning_model && bash scripts/train_finhrm_v14_1_scratch.sh 31 12 cuda"
TRAIN_ONLY SHA-256: eb5ca8c4c53b52d64c2d79533c89a5d4672953247f164cdf57e5561fca6b02e6
Validation SHA-256: 3ddcb5e0a88e431787133b36edacd28a4bbb8b0fd2d1ac29494c0e9ffeec0758
Scratch checkpoint: models/document_hrm_V14_1_bkai_scratch_seed31_v1.pt
Best training state: models/training_state/document_hrm_V14_1_bkai_scratch_seed31_v1.best.train.pt
Last training state: models/training_state/document_hrm_V14_1_bkai_scratch_seed31_v1.last.train.pt
Fresh scratch mode
Command: /usr/bin/python -u main.py --document-model-path models/document_hrm_V14_1_bkai_scratch_seed31_v1.pt train-document-hrm --dataset data/tmp/v14_1_scratch_v1/unified_train.jsonl --validation-dataset data/tmp/v14_development_v1/validation.jsonl --epochs 12 --batch-size 32 --learning-rate 0.0015 --seed 31 --vocab-size 16384 --embedding-dim 64 --hidden-size 96 --high-cycles 1 --low-steps 2 --max-segments 3 --max-candidates 64 --attention-heads 4 --ff-multiplier 3 --synthetic-fraction 1.0 --relevance-prior-max 2.0 --act-mode q_learning --minimum-halt-segments 2 --input-encoder semantic --semantic-model bkai-foundation-models/vietnamese-bi-encoder --semantic-dim 384 --semantic-encoder-backend sentence-transformers --semantic-model-revision 84f9d9ada0d1a3c37557398b9ae9fcedcdf40be0 --semantic-local-files-only --semantic-encoder-dim 768 --semantic-word-segmentation pyvi --semantic-segmenter-version 0.1.1 --semantic-prefix-mode none --semantic-query-encoding-mode query --semantic-text-template finance-vi-v1 --semantic-adapter-mode layernorm-linear-l2-v1 --candidate-semantic-mode structured --financial-structure-mode typed-hierarchy-v1 --financial-hierarchy-layers 1 --variable-sum-mode sum-n-v1 --max-program-operands 8 --program-search-strategy diverse-set-beam-v1 --pointer-conditioning joint-pair --pair-backbone-mode normalized-end-to-end --pair-loss-weight 1.2 --pair-targeted-negative-weight 0.35 --pointer-teacher-forcing 0.5 --pointer-hard-negative-weight 0.2 --pointer-hard-negative-margin 0.5 --semantic-contrastive-weight 0.25 --semantic-contrastive-temperature 0.07 --semantic-contrastive-margin 0.20 --abstention-target-precision 0.90 --abstention-stability-errors 1 --select-best-epoch --best-training-state models/training_state/document_hrm_V14_1_bkai_scratch_seed31_v1.best.train.pt --last-training-state models/training_state/document_hrm_V14_1_bkai_scratch_seed31_v1.last.train.pt --device cuda
Epoch 1/12: training started
Epoch 1/12: batch 1/101, average_loss=17.57649
Epoch 1/12: batch 6/101, average_loss=17.06346
Epoch 1/12: batch 12/101, average_loss=16.46293
Epoch 1/12: batch 18/101, average_loss=15.28641
Epoch 1/12: batch 24/101, average_loss=13.98087
Epoch 1/12: batch 30/101, average_loss=12.97809
Epoch 1/12: batch 36/101, average_loss=12.17833
Epoch 1/12: batch 42/101, average_loss=11.53739
Epoch 1/12: batch 48/101, average_loss=10.95840
Epoch 1/12: batch 54/101, average_loss=10.41533
Epoch 1/12: batch 60/101, average_loss=10.05256
Epoch 1/12: batch 66/101, average_loss=9.86580
Epoch 1/12: batch 72/101, average_loss=9.52072
Epoch 1/12: batch 78/101, average_loss=9.22794
Epoch 1/12: batch 84/101, average_loss=8.96821
Epoch 1/12: batch 90/101, average_loss=8.72975
Epoch 1/12: batch 96/101, average_loss=8.51326
Epoch 1/12: batch 101/101, average_loss=8.32015
Epoch 1/12: loss=8.32015, execution=83.12%, pair=81.60%, program=58.23%, macro_f1=69.42%
Saved best training state: /mnt/d/reasoning_model_V4/reasoning_model/models/training_state/document_hrm_V14_1_bkai_scratch_seed31_v1.best.train.pt
Saved last training state: /mnt/d/reasoning_model_V4/reasoning_model/models/training_state/document_hrm_V14_1_bkai_scratch_seed31_v1.last.train.pt
Epoch 2/12: training started
Epoch 2/12: batch 1/101, average_loss=3.93627
Epoch 2/12: batch 6/101, average_loss=5.17891
Epoch 2/12: batch 12/101, average_loss=5.10911
Epoch 2/12: batch 18/101, average_loss=4.88769
Epoch 2/12: batch 24/101, average_loss=4.70554
Epoch 2/12: batch 30/101, average_loss=4.70882
Epoch 2/12: batch 36/101, average_loss=4.65429
Epoch 2/12: batch 42/101, average_loss=4.63923
Epoch 2/12: batch 48/101, average_loss=4.61985
Epoch 2/12: batch 54/101, average_loss=4.62247
Epoch 2/12: batch 60/101, average_loss=4.62155
Epoch 2/12: batch 66/101, average_loss=4.58632
Epoch 2/12: batch 72/101, average_loss=4.56578
Epoch 2/12: batch 78/101, average_loss=4.52794
Epoch 2/12: batch 84/101, average_loss=4.52219
Epoch 2/12: batch 90/101, average_loss=4.52034
Epoch 2/12: batch 96/101, average_loss=4.50188
Epoch 2/12: batch 101/101, average_loss=4.51654
Epoch 2/12: loss=4.51654, execution=82.25%, pair=80.74%, program=57.36%, macro_f1=68.58%
Saved last training state: /mnt/d/reasoning_model_V4/reasoning_model/models/training_state/document_hrm_V14_1_bkai_scratch_seed31_v1.last.train.pt
Epoch 3/12: training started
Epoch 3/12: batch 1/101, average_loss=4.40967
Epoch 3/12: batch 6/101, average_loss=4.11871
Epoch 3/12: batch 12/101, average_loss=3.70053
Epoch 3/12: batch 18/101, average_loss=3.85425
Epoch 3/12: batch 24/101, average_loss=3.82922
Epoch 3/12: batch 30/101, average_loss=3.82580
Epoch 3/12: batch 36/101, average_loss=3.85094
Epoch 3/12: batch 42/101, average_loss=4.04191
Epoch 3/12: batch 48/101, average_loss=4.02679
Epoch 3/12: batch 54/101, average_loss=4.03318
Epoch 3/12: batch 60/101, average_loss=4.01232
Epoch 3/12: batch 66/101, average_loss=3.96543
Epoch 3/12: batch 72/101, average_loss=4.17626
Epoch 3/12: batch 78/101, average_loss=4.32240
Epoch 3/12: batch 84/101, average_loss=4.33165
Epoch 3/12: batch 90/101, average_loss=4.45709
Epoch 3/12: batch 96/101, average_loss=4.42967
Epoch 3/12: batch 101/101, average_loss=4.50670
Epoch 3/12: loss=4.50670, execution=82.25%, pair=81.17%, program=57.79%, macro_f1=69.13%
Saved last training state: /mnt/d/reasoning_model_V4/reasoning_model/models/training_state/document_hrm_V14_1_bkai_scratch_seed31_v1.last.train.pt
Epoch 4/12: training started
Epoch 4/12: batch 1/101, average_loss=4.84962
Epoch 4/12: batch 6/101, average_loss=3.67699
Epoch 4/12: batch 12/101, average_loss=4.51954
Epoch 4/12: batch 18/101, average_loss=4.49247
Epoch 4/12: batch 24/101, average_loss=4.41106
Epoch 4/12: batch 30/101, average_loss=5.76794
Epoch 4/12: batch 36/101, average_loss=6.09928
Epoch 4/12: batch 42/101, average_loss=5.78474
Epoch 4/12: batch 48/101, average_loss=5.55848
Epoch 4/12: batch 54/101, average_loss=5.35359
Epoch 4/12: batch 60/101, average_loss=5.21438
Epoch 4/12: batch 66/101, average_loss=5.13619
Epoch 4/12: batch 72/101, average_loss=5.02125
Epoch 4/12: batch 78/101, average_loss=4.88282
Epoch 4/12: batch 84/101, average_loss=4.84003
Epoch 4/12: batch 90/101, average_loss=4.74172
Epoch 4/12: batch 96/101, average_loss=4.65811
Epoch 4/12: batch 101/101, average_loss=4.60980
Epoch 4/12: loss=4.60980, execution=82.03%, pair=80.95%, program=56.93%, macro_f1=68.38%
Saved last training state: /mnt/d/reasoning_model_V4/reasoning_model/models/training_state/document_hrm_V14_1_bkai_scratch_seed31_v1.last.train.pt
Epoch 5/12: training started
Epoch 5/12: batch 1/101, average_loss=3.16018
Epoch 5/12: batch 6/101, average_loss=4.56469
Epoch 5/12: batch 12/101, average_loss=4.37937
Epoch 5/12: batch 18/101, average_loss=4.03263
Epoch 5/12: batch 24/101, average_loss=4.10581
Epoch 5/12: batch 30/101, average_loss=4.09209
Epoch 5/12: batch 36/101, average_loss=4.08237
Epoch 5/12: batch 42/101, average_loss=3.97407
Epoch 5/12: batch 48/101, average_loss=3.94188
Epoch 5/12: batch 54/101, average_loss=3.95545
Epoch 5/12: batch 60/101, average_loss=3.91110
Epoch 5/12: batch 66/101, average_loss=3.92944
Epoch 5/12: batch 72/101, average_loss=3.86993
Epoch 5/12: batch 78/101, average_loss=3.90927
Epoch 5/12: batch 84/101, average_loss=3.94496
Epoch 5/12: batch 90/101, average_loss=3.88063
Epoch 5/12: batch 96/101, average_loss=3.81269
Epoch 5/12: batch 101/101, average_loss=3.76074
Epoch 5/12: loss=3.76074, execution=83.77%, pair=82.68%, program=59.31%, macro_f1=69.42%
Saved best training state: /mnt/d/reasoning_model_V4/reasoning_model/models/training_state/document_hrm_V14_1_bkai_scratch_seed31_v1.best.train.pt
Saved last training state: /mnt/d/reasoning_model_V4/reasoning_model/models/training_state/document_hrm_V14_1_bkai_scratch_seed31_v1.last.train.pt
Epoch 6/12: training started
Epoch 6/12: batch 1/101, average_loss=2.54168
Epoch 6/12: batch 6/101, average_loss=2.79135
Epoch 6/12: batch 12/101, average_loss=4.20778
Epoch 6/12: batch 18/101, average_loss=4.34653
Epoch 6/12: batch 24/101, average_loss=4.32691
Epoch 6/12: batch 30/101, average_loss=4.49034
Epoch 6/12: batch 36/101, average_loss=4.53338
Epoch 6/12: batch 42/101, average_loss=4.40512
Epoch 6/12: batch 48/101, average_loss=4.33205
Epoch 6/12: batch 54/101, average_loss=4.23400
Epoch 6/12: batch 60/101, average_loss=4.19472
Epoch 6/12: batch 66/101, average_loss=4.06535
Epoch 6/12: batch 72/101, average_loss=4.32831
Epoch 6/12: batch 78/101, average_loss=4.42139
Epoch 6/12: batch 84/101, average_loss=4.31907
Epoch 6/12: batch 90/101, average_loss=4.22675
Epoch 6/12: batch 96/101, average_loss=4.14671
Epoch 6/12: batch 101/101, average_loss=4.07611
Epoch 6/12: loss=4.07611, execution=80.74%, pair=79.22%, program=55.63%, macro_f1=68.58%
Saved last training state: /mnt/d/reasoning_model_V4/reasoning_model/models/training_state/document_hrm_V14_1_bkai_scratch_seed31_v1.last.train.pt
Epoch 7/12: training started
Epoch 7/12: batch 1/101, average_loss=2.16128
Epoch 7/12: batch 6/101, average_loss=3.91818
Epoch 7/12: batch 12/101, average_loss=3.41029
Epoch 7/12: batch 18/101, average_loss=3.34738
Epoch 7/12: batch 24/101, average_loss=3.11194
Epoch 7/12: batch 30/101, average_loss=3.12040
Epoch 7/12: batch 36/101, average_loss=3.11913
Epoch 7/12: batch 42/101, average_loss=3.26965
Epoch 7/12: batch 48/101, average_loss=3.26805
Epoch 7/12: batch 54/101, average_loss=3.28546
Epoch 7/12: batch 60/101, average_loss=3.37558
Epoch 7/12: batch 66/101, average_loss=3.57181
Epoch 7/12: batch 72/101, average_loss=3.78645
Epoch 7/12: batch 78/101, average_loss=4.55576
Epoch 7/12: batch 84/101, average_loss=4.70925
Epoch 7/12: batch 90/101, average_loss=4.87608
Epoch 7/12: batch 96/101, average_loss=5.23407
Epoch 7/12: batch 101/101, average_loss=5.38707
Epoch 7/12: loss=5.38707, execution=83.33%, pair=82.25%, program=58.66%, macro_f1=69.41%
Saved last training state: /mnt/d/reasoning_model_V4/reasoning_model/models/training_state/document_hrm_V14_1_bkai_scratch_seed31_v1.last.train.pt
Epoch 8/12: training started
Epoch 8/12: batch 1/101, average_loss=6.53450
Epoch 8/12: batch 6/101, average_loss=7.49445
Epoch 8/12: batch 12/101, average_loss=6.88285
Epoch 8/12: batch 18/101, average_loss=6.35275
Epoch 8/12: batch 24/101, average_loss=6.16812
Epoch 8/12: batch 30/101, average_loss=6.03393
Epoch 8/12: batch 36/101, average_loss=6.18093
Epoch 8/12: batch 42/101, average_loss=6.19606
Epoch 8/12: batch 48/101, average_loss=6.38233
Epoch 8/12: batch 54/101, average_loss=6.45307
Epoch 8/12: batch 60/101, average_loss=7.25974
Epoch 8/12: batch 66/101, average_loss=7.34498
Epoch 8/12: batch 72/101, average_loss=7.33841
Epoch 8/12: batch 78/101, average_loss=7.28396
Epoch 8/12: batch 84/101, average_loss=7.17381
Epoch 8/12: batch 90/101, average_loss=7.12080
Epoch 8/12: batch 96/101, average_loss=7.04927
Epoch 8/12: batch 101/101, average_loss=7.00806



Epoch 8/12: loss=7.00806, execution=77.27%, pair=76.41%, program=52.60%, macro_f1=63.61%
Saved last training state: /mnt/d/reasoning_model_V4/reasoning_model/models/training_state/document_hrm_V14_1_bkai_scratch_seed31_v1.last.train.pt
Epoch 9/12: training started
Epoch 9/12: batch 1/101, average_loss=9.12778
Epoch 9/12: batch 6/101, average_loss=6.19133
Epoch 9/12: batch 12/101, average_loss=6.18500
Epoch 9/12: batch 18/101, average_loss=6.27002
Epoch 9/12: batch 24/101, average_loss=6.22609
Epoch 9/12: batch 30/101, average_loss=6.04526
Epoch 9/12: batch 36/101, average_loss=6.10803
Epoch 9/12: batch 42/101, average_loss=5.90533
Epoch 9/12: batch 48/101, average_loss=5.80316
Epoch 9/12: batch 54/101, average_loss=5.77330
Epoch 9/12: batch 60/101, average_loss=5.71245
Epoch 9/12: batch 66/101, average_loss=5.67256
Epoch 9/12: batch 72/101, average_loss=5.58225
Epoch 9/12: batch 78/101, average_loss=5.49992
Epoch 9/12: batch 84/101, average_loss=5.49582
Epoch 9/12: batch 90/101, average_loss=5.46906
Epoch 9/12: batch 96/101, average_loss=5.42963
Epoch 9/12: batch 101/101, average_loss=5.42749
Epoch 9/12: loss=5.42749, execution=77.49%, pair=76.41%, program=52.81%, macro_f1=63.64%
Saved last training state: /mnt/d/reasoning_model_V4/reasoning_model/models/training_state/document_hrm_V14_1_bkai_scratch_seed31_v1.last.train.pt
Epoch 10/12: training started
Epoch 10/12: batch 1/101, average_loss=4.59257
Epoch 10/12: batch 6/101, average_loss=4.82302
Epoch 10/12: batch 12/101, average_loss=4.77156
Epoch 10/12: batch 18/101, average_loss=4.89279
Epoch 10/12: batch 24/101, average_loss=4.81850
Epoch 10/12: batch 30/101, average_loss=4.69855
Epoch 10/12: batch 36/101, average_loss=4.71563
Epoch 10/12: batch 42/101, average_loss=4.57589
Epoch 10/12: batch 48/101, average_loss=4.55025
Epoch 10/12: batch 54/101, average_loss=4.55524
Epoch 10/12: batch 60/101, average_loss=4.57462
Epoch 10/12: batch 66/101, average_loss=4.57721
Epoch 10/12: batch 72/101, average_loss=4.67008
Epoch 10/12: batch 78/101, average_loss=4.63193
Epoch 10/12: batch 84/101, average_loss=4.60421
Epoch 10/12: batch 90/101, average_loss=4.56965
Epoch 10/12: batch 96/101, average_loss=4.51646
Epoch 10/12: batch 101/101, average_loss=4.51506


Epoch 10/12: loss=4.51506, execution=78.57%, pair=77.49%, program=53.90%, macro_f1=65.42%
Saved last training state: /mnt/d/reasoning_model_V4/reasoning_model/models/training_state/document_hrm_V14_1_bkai_scratch_seed31_v1.last.train.pt
Epoch 11/12: training started
Epoch 11/12: batch 1/101, average_loss=4.43062
Epoch 11/12: batch 6/101, average_loss=3.72733
Epoch 11/12: batch 12/101, average_loss=3.69613
Epoch 11/12: batch 18/101, average_loss=3.72411
Epoch 11/12: batch 24/101, average_loss=3.68307
Epoch 11/12: batch 30/101, average_loss=3.77246
Epoch 11/12: batch 36/101, average_loss=3.94551
Epoch 11/12: batch 42/101, average_loss=3.93698
Epoch 11/12: batch 48/101, average_loss=4.04106
Epoch 11/12: batch 54/101, average_loss=4.06336
Epoch 11/12: batch 60/101, average_loss=4.13827
Epoch 11/12: batch 66/101, average_loss=4.13443
Epoch 11/12: batch 72/101, average_loss=4.13874
Epoch 11/12: batch 78/101, average_loss=4.12664
Epoch 11/12: batch 84/101, average_loss=4.14203
Epoch 11/12: batch 90/101, average_loss=4.14612
Epoch 11/12: batch 96/101, average_loss=4.12688
Epoch 11/12: batch 101/101, average_loss=4.08912
Epoch 11/12: loss=4.08912, execution=80.30%, pair=79.22%, program=55.84%, macro_f1=69.35%
Saved last training state: /mnt/d/reasoning_model_V4/reasoning_model/models/training_state/document_hrm_V14_1_bkai_scratch_seed31_v1.last.train.pt
Epoch 12/12: training started
Epoch 12/12: batch 1/101, average_loss=3.25725
Epoch 12/12: batch 6/101, average_loss=4.35793
Epoch 12/12: batch 12/101, average_loss=3.89240
Epoch 12/12: batch 18/101, average_loss=3.88049
Epoch 12/12: batch 24/101, average_loss=3.89585
Epoch 12/12: batch 30/101, average_loss=3.92438
Epoch 12/12: batch 36/101, average_loss=3.92141
Epoch 12/12: batch 42/101, average_loss=3.88499
Epoch 12/12: batch 48/101, average_loss=4.04815
Epoch 12/12: batch 54/101, average_loss=3.98232
Epoch 12/12: batch 60/101, average_loss=3.97527
Epoch 12/12: batch 66/101, average_loss=3.94766
Epoch 12/12: batch 72/101, average_loss=3.96248
Epoch 12/12: batch 78/101, average_loss=3.99453
Epoch 12/12: batch 84/101, average_loss=3.95544
Epoch 12/12: batch 90/101, average_loss=3.92934
Epoch 12/12: batch 96/101, average_loss=3.89855
Epoch 12/12: batch 101/101, average_loss=3.88375
Epoch 12/12: loss=3.88375, execution=80.95%, pair=79.87%, program=56.49%, macro_f1=69.66%
Saved last training state: /mnt/d/reasoning_model_V4/reasoning_model/models/training_state/document_hrm_V14_1_bkai_scratch_seed31_v1.last.train.pt
Selected best epoch: 5/12 (execution=83.77%, pair=82.68%, program=59.31%, macro_f1=69.42%)













Saved Document HRM: /mnt/d/reasoning_model_V4/reasoning_model/models/document_hrm_V14_1_bkai_scratch_seed31_v1.pt
Trainable parameters: 1,388,324
Train=2167, validation=462, epochs=12, loss=3.76074
Operation accuracy: 98.48%
Operation macro P/R/F1: 99.04% / 99.55% / 99.27%
Operation weighted P/R/F1: 99.20% / 99.13% / 99.14%
Program exact accuracy: 82.68%
Execution accuracy: 83.77%
Pointer pair accuracy: 82.68%
Pointer slot accuracy: p1=84.63%, p2=81.48%, p3=90.00%, p4=90.00%, p5=0.00%, p6=0.00%, p7=0.00%, p8=0.00%
Accepted coverage/precision: 0.00% / 0.00%
(base) PS D:\reasoning_model_V4\reasoning_model> 