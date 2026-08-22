# strict 4+2 逐类场景软门控 dev 搜索

线上门控只读取 Scene-SensorNet 输出的已知场景概率，不读取文件名或真值标签。
Base 类先验只由 base train 正样本学习；新增类先验只由 increment train 正样本学习。

| 候选 | Base mAP50 | New-mAP50 | KRR | 六类 precision | 六类 FP | 新类 precision | 新类 FP | 误激活图像 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.909986 | 0.829740 | 1.000000 | 0.271622 | 1078 | 0.367521 | 148 | 72 |
| score_first | 0.910667 | 0.910558 | 1.001804 | 0.410733 | 571 | 0.677686 | 39 | 33 |
| guarded_precision | 0.889979 | 0.780844 | 0.980132 | 0.740594 | 131 | 0.958904 | 3 | 6 |
| balanced_precision | 0.873509 | 0.780844 | 0.961391 | 0.717349 | 145 | 0.958904 | 3 | 5 |

## 候选参数

### score_first

- 阈值：`{'0': 0.01, '1': 0.09, '2': 0.36, '3': 0.01, '4': 0.33, '5': 0.02}`
- 最大场景惩罚：`{'0': 0.13, '1': 0.8, '2': 0.26, '3': 0.32, '4': 0.7, '5': 0.72}`
- 跨类冲突：`{'enabled': True, 'iou': 0.3, 'base_confidence': 0.7, 'specialist_margin': 0.0, 'preserve_base_class_owners': True}`

### guarded_precision

- 阈值：`{'0': 0.21, '1': 0.14, '2': 0.36, '3': 0.05, '4': 0.57, '5': 0.82}`
- 最大场景惩罚：`{'0': 0.15, '1': 0.88, '2': 0.26, '3': 0.19, '4': 0.65, '5': 0.0}`
- 跨类冲突：`{'enabled': False, 'iou': 1.0, 'base_confidence': 0.01, 'specialist_margin': 0.0, 'preserve_base_class_owners': True}`

### balanced_precision

- 阈值：`{'0': 0.29, '1': 0.14, '2': 0.36, '3': 0.01, '4': 0.57, '5': 0.82}`
- 最大场景惩罚：`{'0': 0.15, '1': 0.88, '2': 0.26, '3': 0.32, '4': 0.65, '5': 0.0}`
- 跨类冲突：`{'enabled': False, 'iou': 1.0, 'base_confidence': 0.01, 'specialist_margin': 0.0, 'preserve_base_class_owners': True}`

