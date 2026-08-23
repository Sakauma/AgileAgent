# 舰船 3+1 复现实验

该实验使用 soldier、small_aircraft、tank 作为基础类别，warship 作为增量类别，验证三类冻结基础检测器与单类增量检测器的组合。

## 数据

| 清单 | 图片数 |
| --- | ---: |
| `base_train.txt` | 441 |
| `base_dev.txt` | 70 |
| `increment_train.txt` | 132 |
| `increment_dev.txt` | 18 |
| `base_test.txt` | 70 |
| `mixed_test.txt` | 89 |

## 运行

```bash
agile-agent experiment validate \
  --config configs/incremental/warship_3plus1.yaml

agile-agent experiment run \
  --config configs/incremental/warship_3plus1.yaml
```

复现已生成实验：

```bash
agile-agent experiment reproduce \
  --manifest runs/experiments/warship_3plus1/RUN_ID/run_manifest.json
```

## 当前 production 结果

| 指标 | 数值 |
| --- | ---: |
| Base mAP50 | `0.814142` |
| New-mAP50 | `0.638688` |
| KRR | `1.000000` |
| 新类 precision | `0.924528` |
| 误激活率 | `0.014286` |

实验 manifest 记录数据清单 SHA256、配置、父代际、模型权重、阈值、类别映射、环境和逐图指标。当前 production 将基础类别所有权绑定到三类基础检测器，将全局类别 ID `2` 绑定到增量检测器。

多轮机制已完成四批次回归和 balanced micro、sensor shift、diminishing 三组共 21 轮压力回归。
