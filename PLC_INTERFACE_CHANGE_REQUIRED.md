# PLC联调接口变更（必须先改PLC，否则此服务不能投产）

本目录是基于上传的 **实际运行版 VisionService.zip** 修改的 Python 候选版；不是已完成PLC联调的最终版本。

## 新增 ADS 字段

在 PLC `GVL_ScrewVision.stInterface` 对应结构体增加：

```iecst
sPresenceResult : STRING(1); // 'N'=无ZNQ/跳点; 'Y'=有ZNQ/执行螺钉检测; ''=尚无有效判断
fEmptyConfidence : LREAL; // Circle最高置信度，用于调试
```

请确认原 PLC 程序所用变量名称、类型，并提供源代码后再最终确定接口（这里是提议字段）。

## 一笔事务（同一张ME2P帧）

1. PLC 发 Screw `RequestId`；同一 worker 获取 **一帧**。
2. 第一模型 `empty_slot.pt` 检测 Circle，最大置信度 >= 0.70 => `N`，**不调用螺钉模型**。
3. 没检测到符合阈值的 Circle => `Y`，同一帧送到原 `best.pt` ScrewSpringDetector。
4. 对 `Y`，如果 Screw 少于2颗或原算法无效：`bResultInvalid=TRUE`，PLC进入故障，绝不能跳过。
5. 对 `N`，`bResultInvalid=FALSE`、`bDetected=FALSE`、角度=0，但**只有 `sPresenceResult='N'` 可以跳点**。
6. 对 `Y` 且 Screw有效：按原逻辑输出角度。
7. Python最后写 `bResultValid=TRUE`；PLC同时校验 ResultId 与 RequestId。
8. 相机故障/模型异常/ADS通讯失败：不得发布 N；设备侧必须按超时或故障策略处理。

## 必须删除/禁用的 PLC 旧逻辑

- 禁止通过 `NOT bDetected` / Screw不足推断正常空位、跳过点位。
- 删除或重新路由 `SkippedNoScrew` 等分支：只有已提交有效结果，且 `sPresenceResult='N'`，才允许跳点。
- `sPresenceResult='Y'` 且 `bResultInvalid=TRUE`，必须进入故障，不能跳点。
- 对 ADS旧 RequestId、空字符串、其他字符串、超时，按故障处理。

## 后续还需要上传

**当前工控机运行的完整 TwinCAT 工程 ZIP（不是旧版备份）**。至少包括：
- `GVL_ScrewVision` 和其接口 `ST_ScrewVisionInterface`；
- `FB_VisionRequest`、`FB_ScrewAngleVision`；
- `FB_DamperProcess`、工位/点位循环的状态机及枚举；
- 如果 WPF 参与跳点或状态显示，还需实际 WPF 的 ViewModel/ADS 通讯实现。

## 未完成实机验证

本环境未连接 PLC、大恒相机，也没有完成现场 HALCON/YOLO 实际推理。必须通过真空位、有料、遮挡、空位误判、单螺钉、相机掉线及ADS断线测试后才能用于自动运行。

> 特别提醒：当前映射是按需求确定的 **检测到 Circle = 空位 N**；这不是根据模型类别名称就能独立证明的物理含义，务必用实际标注样图验证。没有检测到Circle的 `Y` 并不代表模型绝对可靠；低置信度有料/空位需要现场验证误判风险。
